"""RQ tasks for hosted ingestion and evaluation workloads."""

from __future__ import annotations

import json

import httpx
from sqlalchemy import func, select, text, update

from apps.api.main import (
    CorpusElement,
    DatasetRevision,
    Document,
    EvaluationRun,
    Experiment,
    Job,
    SessionLocal,
)
from apps.worker.pg_queue import StaleLeaseError, complete_job
from core.config import get_settings
from core.generate.pipeline import GroundedAnswerGenerator
from core.llm.router import get_chat_provider
from core.normalize.schema import Chunk
from core.retrieval.schema import RetrievalStrategy, SearchResult
from eval.datasets.schema import EvaluationDataset
from eval.runners.evaluate import EvaluationRunner
from eval.runners.experiment import (
    ExperimentConfig,
    ExperimentRunner,
    ExperimentVariant,
)
from eval.runners.run_config import EvaluationRunConfig


class HostedElementRetriever:
    """Exact cosine retrieval over workspace-scoped source elements."""

    def __init__(self, elements: list[CorpusElement], max_depth: int = 20) -> None:
        self.elements = elements
        self.max_depth = max_depth
        settings = get_settings()
        self.model_id = settings.embedding_model
        self.base_url = settings.embedding_base_url.rstrip("/")

    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        response = httpx.post(
            f"{self.base_url}/api/embed", json={"model": self.model_id, "input": query}, timeout=90
        )
        response.raise_for_status()
        query_vector = response.json()["embeddings"][0]
        scored = []
        for element in self.elements:
            vector = json.loads(element.embedding_json or "[]")
            if len(vector) != len(query_vector) or not vector:
                continue
            denom = (sum(v * v for v in vector) * sum(v * v for v in query_vector)) ** 0.5
            score = (
                sum(a * b for a, b in zip(vector, query_vector, strict=True)) / denom
                if denom
                else 0.0
            )
            scored.append((score, element))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [
            SearchResult(
                chunk=Chunk(
                    chunk_id=element.id,
                    document_id=element.document_id,
                    text=element.text,
                    chunk_index=element.ordinal,
                    source_element_ids=[element.id],
                ),
                score=score,
                rank=rank,
                strategy=RetrievalStrategy.DENSE,
            )
            for rank, (score, element) in enumerate(scored[: min(top_k, self.max_depth)])
        ]


def index_document(
    workspace_id: str,
    document_id: str,
    job_id: str,
    lease_token: str | None = None,
    lease_generation: int | None = None,
    worker_id: str | None = None,
    attempt_id: str | None = None,
    session_factory=None,
) -> dict[str, int]:
    """Embed source elements and persist vectors using 3-phase fenced transactions."""
    sf = session_factory or SessionLocal

    # -------------------------------------------------------------
    # Phase A: Short DB read (<50ms)
    # -------------------------------------------------------------
    with sf() as db:
        is_pg = bool(db.bind and db.bind.dialect.name == "postgresql")
        if is_pg:
            db.execute(
                text("SELECT set_config('app.workspace_id', :workspace_id, true)"),
                {"workspace_id": workspace_id},
            )
            if not (lease_token and lease_generation is not None and worker_id and attempt_id):
                raise StaleLeaseError(
                    "PostgreSQL worker execution strictly requires active lease_token, "
                    "lease_generation, worker_id, and attempt_id."
                )

        doc = db.scalar(
            select(Document).where(
                Document.id == document_id, Document.workspace_id == workspace_id
            )
        )
        job = db.scalar(select(Job).where(Job.id == job_id, Job.workspace_id == workspace_id))
        if doc is None or job is None:
            return {"indexed": 0}

        if lease_token:
            # Verify active lease
            active = db.scalar(
                select(Job.id).where(
                    Job.id == job_id,
                    Job.workspace_id == workspace_id,
                    Job.lease_token == lease_token,
                    Job.lease_generation == lease_generation,
                    Job.worker_id == worker_id,
                    Job.status == "running",
                    Job.lease_expires_at > func.now(),
                )
            )
            if not active:
                raise StaleLeaseError("Lease expired before index task execution.")
        else:
            doc.status = "running"
            job.status = "running"
            db.commit()

        elements = list(
            db.scalars(
                select(CorpusElement)
                .where(
                    CorpusElement.workspace_id == workspace_id,
                    CorpusElement.document_id == document_id,
                )
                .order_by(CorpusElement.ordinal)
            )
        )
        total_elements = len(elements)
        if total_elements == 0:
            if not lease_token:
                doc.status = "failed"
                job.status = "failed"
                job.detail = "Cannot index empty corpus: document has zero source elements."
                db.commit()
            raise ValueError("Cannot index empty corpus: document has zero source elements.")

        missing_elements = [(el.id, el.text) for el in elements if not el.embedding_json]

    # -------------------------------------------------------------
    # Phase B: External I/O without holding database connection
    # -------------------------------------------------------------
    try:
        settings = get_settings()
        base = settings.embedding_base_url.rstrip("/")
        model = settings.embedding_model
        texts = [text for _, text in missing_elements]
        vectors = []
        if texts:
            response = httpx.post(
                f"{base}/api/embed", json={"model": model, "input": texts}, timeout=180
            )
            response.raise_for_status()
            vectors = response.json().get("embeddings", [])
            if len(vectors) != len(texts):
                raise ValueError("Embedding service returned an unexpected vector count.")
    except Exception as exc:
        if not lease_token:
            with sf() as db:
                db.rollback()
                if db.bind and db.bind.dialect.name == "postgresql":
                    db.execute(
                        text("SELECT set_config('app.workspace_id', :wid, true)"),
                        {"wid": workspace_id},
                    )
                doc = db.scalar(
                    select(Document).where(
                        Document.id == document_id, Document.workspace_id == workspace_id
                    )
                )
                job = db.scalar(
                    select(Job).where(Job.id == job_id, Job.workspace_id == workspace_id)
                )
                if doc:
                    doc.status = "failed"
                if job:
                    job.status = "failed"
                    job.detail = f"Indexing failed: {exc}"
                db.commit()
        raise

    # -------------------------------------------------------------
    # Phase C: Atomic fenced write (<50ms)
    # -------------------------------------------------------------
    with sf() as db:
        if db.bind and db.bind.dialect.name == "postgresql":
            db.execute(
                text("SELECT set_config('app.workspace_id', :workspace_id, true)"),
                {"workspace_id": workspace_id},
            )
        if lease_token:
            locked = db.scalar(
                select(Job.id)
                .where(
                    Job.id == job_id,
                    Job.workspace_id == workspace_id,
                    Job.lease_token == lease_token,
                    Job.lease_generation == lease_generation,
                    Job.worker_id == worker_id,
                    Job.status == "running",
                    Job.lease_expires_at > func.now(),
                )
                .with_for_update()
            )
            if not locked:
                db.rollback()
                raise StaleLeaseError(
                    "Fencing check failed: lease expired or lost during indexing."
                )

        for (el_id, _), vec in zip(missing_elements, vectors, strict=True):
            db.execute(
                update(CorpusElement)
                .where(CorpusElement.id == el_id, CorpusElement.workspace_id == workspace_id)
                .values(embedding_json=json.dumps(vec, separators=(",", ":")))
            )

        db.execute(
            update(Document)
            .where(Document.id == document_id, Document.workspace_id == workspace_id)
            .values(status="indexed")
        )

        detail_msg = f"{total_elements} source elements indexed"
        if lease_token and attempt_id and lease_generation is not None and worker_id is not None:
            ok = complete_job(
                db,
                job_id=job_id,
                lease_token=lease_token,
                lease_generation=lease_generation,
                worker_id=worker_id,
                attempt_id=attempt_id,
                detail=detail_msg,
            )
            if not ok:
                db.rollback()
                raise StaleLeaseError(
                    "Fencing check failed: lease invalidated during job completion."
                )
        else:
            db.execute(
                update(Job)
                .where(Job.id == job_id, Job.workspace_id == workspace_id)
                .values(status="complete", detail=detail_msg)
            )
        db.commit()
        return {"indexed": total_elements}


def evaluate_dataset(
    workspace_id: str,
    run_id: str,
    job_id: str,
    revision_id: str,
    cutoffs: list[int],
    lease_token: str | None = None,
    lease_generation: int | None = None,
    worker_id: str | None = None,
    attempt_id: str | None = None,
    session_factory=None,
) -> dict[str, str]:
    """Run retrieval evaluation asynchronously using 3-phase fenced transactions."""
    sf = session_factory or SessionLocal

    # -------------------------------------------------------------
    # Phase A: Short DB read (<50ms)
    # -------------------------------------------------------------
    with sf() as db:
        is_pg = bool(db.bind and db.bind.dialect.name == "postgresql")
        if is_pg:
            db.execute(
                text("SELECT set_config('app.workspace_id', :wid, true)"), {"wid": workspace_id}
            )
            if not (lease_token and lease_generation is not None and worker_id and attempt_id):
                raise StaleLeaseError(
                    "PostgreSQL worker execution strictly requires active lease_token, "
                    "lease_generation, worker_id, and attempt_id."
                )
        run = db.scalar(
            select(EvaluationRun).where(
                EvaluationRun.id == run_id, EvaluationRun.workspace_id == workspace_id
            )
        )
        job = db.scalar(select(Job).where(Job.id == job_id, Job.workspace_id == workspace_id))
        revision = db.scalar(
            select(DatasetRevision).where(
                DatasetRevision.id == revision_id,
                DatasetRevision.workspace_id == workspace_id,
            )
        )
        elements = list(
            db.scalars(
                select(CorpusElement).where(
                    CorpusElement.workspace_id == workspace_id,
                    CorpusElement.embedding_json.is_not(None),
                )
            )
        )
        if run is None or job is None or revision is None:
            raise LookupError("Evaluation run, job, or dataset revision is unavailable.")

        if lease_token:
            active = db.scalar(
                select(Job.id).where(
                    Job.id == job_id,
                    Job.workspace_id == workspace_id,
                    Job.lease_token == lease_token,
                    Job.lease_generation == lease_generation,
                    Job.worker_id == worker_id,
                    Job.status == "running",
                    Job.lease_expires_at > func.now(),
                )
            )
            if not active:
                raise StaleLeaseError("Lease expired before evaluation task execution.")
        else:
            run.status = "running"
            job.status = "running"
            db.commit()

        cases = json.loads(revision.cases_json)
        ds_id = f"{revision.dataset_id}:r{revision.revision}"
        sha256 = revision.sha256

    # -------------------------------------------------------------
    # Phase B: External I/O without holding database connection
    # -------------------------------------------------------------
    try:
        dataset = EvaluationDataset(
            dataset_id=ds_id,
            sha256=sha256,
            cases=cases,
        )
        settings = get_settings()
        config = EvaluationRunConfig(cutoffs=tuple(cutoffs))
        retriever = HostedElementRetriever(elements)
        answerer = GroundedAnswerGenerator(
            retriever,
            get_chat_provider(settings.chat_provider),
            top_k=config.max_cutoff,
        )
        report = EvaluationRunner(config).run(dataset, retriever, answerer)
    except Exception as exc:
        if not lease_token:
            with sf() as db:
                db.rollback()
                if db.bind and db.bind.dialect.name == "postgresql":
                    db.execute(
                        text("SELECT set_config('app.workspace_id', :wid, true)"),
                        {"wid": workspace_id},
                    )
                run = db.scalar(
                    select(EvaluationRun).where(
                        EvaluationRun.id == run_id, EvaluationRun.workspace_id == workspace_id
                    )
                )
                job = db.scalar(
                    select(Job).where(Job.id == job_id, Job.workspace_id == workspace_id)
                )
                if run:
                    run.status = "failed"
                if job:
                    job.status = "failed"
                    job.detail = f"Evaluation failed: {type(exc).__name__}"
                db.commit()
        raise

    # -------------------------------------------------------------
    # Phase C: Atomic fenced write (<50ms)
    # -------------------------------------------------------------
    with sf() as db:
        if db.bind and db.bind.dialect.name == "postgresql":
            db.execute(
                text("SELECT set_config('app.workspace_id', :wid, true)"), {"wid": workspace_id}
            )
        if lease_token:
            locked = db.scalar(
                select(Job.id)
                .where(
                    Job.id == job_id,
                    Job.workspace_id == workspace_id,
                    Job.lease_token == lease_token,
                    Job.lease_generation == lease_generation,
                    Job.worker_id == worker_id,
                    Job.status == "running",
                    Job.lease_expires_at > func.now(),
                )
                .with_for_update()
            )
            if not locked:
                db.rollback()
                raise StaleLeaseError(
                    "Fencing check failed: lease expired or lost during evaluation."
                )

        db.execute(
            update(EvaluationRun)
            .where(EvaluationRun.id == run_id, EvaluationRun.workspace_id == workspace_id)
            .values(status=report.status, report_json=report.model_dump_json())
        )

        detail_msg = f"{len(report.queries)} queries evaluated"
        final_job_status = "complete" if report.status == "complete" else report.status
        if lease_token and attempt_id and lease_generation is not None and worker_id is not None:
            ok = complete_job(
                db,
                job_id=job_id,
                lease_token=lease_token,
                lease_generation=lease_generation,
                worker_id=worker_id,
                attempt_id=attempt_id,
                detail=detail_msg,
            )
            if not ok:
                db.rollback()
                raise StaleLeaseError(
                    "Fencing check failed: lease invalidated during evaluation completion."
                )
        else:
            db.execute(
                update(Job)
                .where(Job.id == job_id, Job.workspace_id == workspace_id)
                .values(status=final_job_status, detail=detail_msg)
            )
        db.commit()
        return {"run_id": run_id, "status": report.status}


def run_experiment(
    workspace_id: str,
    experiment_id: str,
    job_id: str,
    revision_id: str,
    payload: dict,
    lease_token: str | None = None,
    lease_generation: int | None = None,
    worker_id: str | None = None,
    attempt_id: str | None = None,
    session_factory=None,
) -> dict[str, str]:
    """Compare named retrieval-depth variants using 3-phase fenced transactions."""
    sf = session_factory or SessionLocal

    # -------------------------------------------------------------
    # Phase A: Short DB read (<50ms)
    # -------------------------------------------------------------
    with sf() as db:
        is_pg = bool(db.bind and db.bind.dialect.name == "postgresql")
        if is_pg:
            db.execute(
                text("SELECT set_config('app.workspace_id', :wid, true)"), {"wid": workspace_id}
            )
            if not (lease_token and lease_generation is not None and worker_id and attempt_id):
                raise StaleLeaseError(
                    "PostgreSQL worker execution strictly requires active lease_token, "
                    "lease_generation, worker_id, and attempt_id."
                )
        experiment = db.scalar(
            select(Experiment).where(
                Experiment.id == experiment_id,
                Experiment.workspace_id == workspace_id,
            )
        )
        job = db.scalar(select(Job).where(Job.id == job_id, Job.workspace_id == workspace_id))
        revision = db.scalar(
            select(DatasetRevision).where(
                DatasetRevision.id == revision_id,
                DatasetRevision.workspace_id == workspace_id,
            )
        )
        if experiment is None or job is None or revision is None:
            raise LookupError("Experiment, job, or dataset revision is unavailable.")

        if lease_token:
            active = db.scalar(
                select(Job.id).where(
                    Job.id == job_id,
                    Job.workspace_id == workspace_id,
                    Job.lease_token == lease_token,
                    Job.lease_generation == lease_generation,
                    Job.worker_id == worker_id,
                    Job.status == "running",
                    Job.lease_expires_at > func.now(),
                )
            )
            if not active:
                raise StaleLeaseError("Lease expired before experiment task execution.")
        else:
            experiment.status = "running"
            job.status = "running"
            db.commit()

        elements = list(
            db.scalars(
                select(CorpusElement).where(
                    CorpusElement.workspace_id == workspace_id,
                    CorpusElement.embedding_json.is_not(None),
                )
            )
        )
        ds_id = f"{revision.dataset_id}:r{revision.revision}"
        sha256 = revision.sha256
        cases = json.loads(revision.cases_json)

    # -------------------------------------------------------------
    # Phase B: External I/O without holding database connection
    # -------------------------------------------------------------
    try:
        dataset = EvaluationDataset(
            dataset_id=ds_id,
            sha256=sha256,
            cases=cases,
        )
        evaluation_config = EvaluationRunConfig(cutoffs=tuple(payload["cutoffs"]))
        config = ExperimentConfig(
            experiment_id=experiment_id,
            baseline_variant=payload["baseline_variant"],
            evaluation=evaluation_config,
            bootstrap_resamples=payload["bootstrap_resamples"],
            random_seed=payload["random_seed"],
        )
        variants = []
        for item in payload["variants"]:
            retriever = HostedElementRetriever(elements, item["retrieval_depth"])
            settings = get_settings()
            answerer = GroundedAnswerGenerator(
                retriever,
                get_chat_provider(settings.chat_provider),
                top_k=evaluation_config.max_cutoff,
            )
            variants.append(
                ExperimentVariant(
                    name=item["name"],
                    pipeline_config={"retrieval_depth": item["retrieval_depth"]},
                    retriever=retriever,
                    answerer=answerer,
                )
            )
        report = ExperimentRunner(config).run(dataset, variants)
    except Exception:
        if not lease_token:
            with sf() as db:
                db.rollback()
                if db.bind and db.bind.dialect.name == "postgresql":
                    db.execute(
                        text("SELECT set_config('app.workspace_id', :wid, true)"),
                        {"wid": workspace_id},
                    )
                experiment = db.scalar(
                    select(Experiment).where(
                        Experiment.id == experiment_id,
                        Experiment.workspace_id == workspace_id,
                    )
                )
                job = db.scalar(
                    select(Job).where(Job.id == job_id, Job.workspace_id == workspace_id)
                )
                if experiment:
                    experiment.status = "failed"
                if job:
                    job.status = "failed"
                    job.detail = (
                        "Experiment failed; inspect worker logs for sanitized exception type."
                    )
                db.commit()
        raise

    # -------------------------------------------------------------
    # Phase C: Atomic fenced write (<50ms)
    # -------------------------------------------------------------
    with sf() as db:
        if db.bind and db.bind.dialect.name == "postgresql":
            db.execute(
                text("SELECT set_config('app.workspace_id', :wid, true)"), {"wid": workspace_id}
            )
        if lease_token:
            locked = db.scalar(
                select(Job.id)
                .where(
                    Job.id == job_id,
                    Job.workspace_id == workspace_id,
                    Job.lease_token == lease_token,
                    Job.lease_generation == lease_generation,
                    Job.worker_id == worker_id,
                    Job.status == "running",
                    Job.lease_expires_at > func.now(),
                )
                .with_for_update()
            )
            if not locked:
                db.rollback()
                raise StaleLeaseError(
                    "Fencing check failed: lease expired or lost during experiment."
                )

        db.execute(
            update(Experiment)
            .where(Experiment.id == experiment_id, Experiment.workspace_id == workspace_id)
            .values(status=report.status, report_json=report.model_dump_json())
        )

        detail_msg = f"{len(report.variants)} variants compared"
        final_job_status = "complete" if report.status == "complete" else report.status
        if lease_token and attempt_id and lease_generation is not None and worker_id is not None:
            ok = complete_job(
                db,
                job_id=job_id,
                lease_token=lease_token,
                lease_generation=lease_generation,
                worker_id=worker_id,
                attempt_id=attempt_id,
                detail=detail_msg,
            )
            if not ok:
                db.rollback()
                raise StaleLeaseError(
                    "Fencing check failed: lease invalidated during experiment completion."
                )
        else:
            db.execute(
                update(Job)
                .where(Job.id == job_id, Job.workspace_id == workspace_id)
                .values(status=final_job_status, detail=detail_msg)
            )
        db.commit()
        return {"experiment_id": experiment_id, "status": report.status}
