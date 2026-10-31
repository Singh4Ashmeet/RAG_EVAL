"""RQ tasks for hosted ingestion and evaluation workloads."""

from __future__ import annotations

import json
import os

import httpx
from sqlalchemy import select, text

from apps.api.main import (
    PRIVATE_DATA,
    CorpusElement,
    DatasetRevision,
    Document,
    EvaluationRun,
    Experiment,
    Job,
    SessionLocal,
)
from core.generate.pipeline import GroundedAnswerGenerator
from core.llm.ollama_provider import OllamaLLMProvider
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
        self.model_id = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        response = httpx.post(
            f"{base}/api/embed", json={"model": self.model_id, "input": query}, timeout=90
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


def index_document(workspace_id: str, document_id: str, job_id: str) -> dict[str, int]:
    """Embed source elements and persist vectors under their owning workspace."""
    with SessionLocal() as db:
        if db.bind and db.bind.dialect.name == "postgresql":
            db.execute(
                text("SELECT set_config('app.workspace_id', :workspace_id, true)"),
                {"workspace_id": workspace_id},
            )
        doc = db.scalar(
            select(Document).where(
                Document.id == document_id, Document.workspace_id == workspace_id
            )
        )
        job = db.scalar(select(Job).where(Job.id == job_id, Job.workspace_id == workspace_id))
        if doc is None or job is None:
            return {"indexed": 0}
        doc.status = "running"
        job.status = "running"
        db.commit()
        try:
            if db.bind and db.bind.dialect.name == "postgresql":
                db.execute(
                    text("SELECT set_config('app.workspace_id', :wid, true)"), {"wid": workspace_id}
                )
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
            # Verify the private original is still present before indexing its derived elements.
            if os.getenv("S3_BUCKET"):
                import boto3
                from botocore.config import Config

                endpoint = os.getenv("S3_ENDPOINT_URL")
                options = {
                    "endpoint_url": endpoint,
                    "region_name": os.getenv("AWS_REGION", "us-east-1"),
                }
                if endpoint:
                    options["config"] = Config(s3={"addressing_style": "path"})
                boto3.client("s3", **options).head_object(
                    Bucket=os.environ["S3_BUCKET"], Key=doc.storage_key
                )
            elif not (PRIVATE_DATA / doc.storage_key).is_file():
                raise FileNotFoundError("Original source file no longer exists.")
            base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
            texts = [element.text for element in elements]
            if texts:
                response = httpx.post(
                    f"{base}/api/embed", json={"model": model, "input": texts}, timeout=180
                )
                response.raise_for_status()
                vectors = response.json().get("embeddings", [])
                if len(vectors) != len(elements):
                    raise ValueError("Embedding service returned an unexpected vector count.")
                for element, vector in zip(elements, vectors, strict=True):
                    element.embedding_json = json.dumps(vector, separators=(",", ":"))
            doc.status = "indexed"
            job.status = "complete"
            job.detail = f"{len(elements)} source elements indexed"
            db.commit()
            return {"indexed": len(elements)}
        except Exception as exc:
            doc.status = "failed"
            job.status = "failed"
            job.detail = f"Indexing failed: {type(exc).__name__}"
            db.commit()
            raise


def evaluate_dataset(
    workspace_id: str,
    run_id: str,
    job_id: str,
    revision_id: str,
    cutoffs: list[int],
) -> dict[str, str]:
    """Run retrieval evaluation asynchronously and persist its report."""
    with SessionLocal() as db:
        if db.bind and db.bind.dialect.name == "postgresql":
            db.execute(
                text("SELECT set_config('app.workspace_id', :wid, true)"), {"wid": workspace_id}
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
        run.status = "running"
        job.status = "running"
        db.commit()
        try:
            # Reapply transaction-local tenant context after the state commit.
            if db.bind and db.bind.dialect.name == "postgresql":
                db.execute(
                    text("SELECT set_config('app.workspace_id', :wid, true)"), {"wid": workspace_id}
                )
            elements = list(
                db.scalars(
                    select(CorpusElement).where(
                        CorpusElement.workspace_id == workspace_id,
                        CorpusElement.embedding_json.is_not(None),
                    )
                )
            )
            dataset = EvaluationDataset(
                dataset_id=f"{revision.dataset_id}:r{revision.revision}",
                sha256=revision.sha256,
                cases=json.loads(revision.cases_json),
            )
            config = EvaluationRunConfig(cutoffs=tuple(cutoffs))
            retriever = HostedElementRetriever(elements)
            answerer = GroundedAnswerGenerator(
                retriever,
                OllamaLLMProvider(
                    model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b"),
                    host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                ),
                top_k=config.max_cutoff,
            )
            report = EvaluationRunner(config).run(dataset, retriever, answerer)
            if db.bind and db.bind.dialect.name == "postgresql":
                db.execute(
                    text("SELECT set_config('app.workspace_id', :wid, true)"), {"wid": workspace_id}
                )
            run = db.scalar(
                select(EvaluationRun).where(
                    EvaluationRun.id == run_id,
                    EvaluationRun.workspace_id == workspace_id,
                )
            )
            job = db.scalar(select(Job).where(Job.id == job_id, Job.workspace_id == workspace_id))
            if run is None or job is None:
                raise LookupError("Evaluation state disappeared during execution.")
            run.status = report.status
            run.report_json = report.model_dump_json()
            job.status = "complete" if report.status == "complete" else report.status
            job.detail = f"{len(report.queries)} queries evaluated"
            db.commit()
            return {"run_id": run_id, "status": report.status}
        except Exception as exc:
            db.rollback()
            if db.bind and db.bind.dialect.name == "postgresql":
                db.execute(
                    text("SELECT set_config('app.workspace_id', :wid, true)"), {"wid": workspace_id}
                )
            run = db.scalar(
                select(EvaluationRun).where(
                    EvaluationRun.id == run_id,
                    EvaluationRun.workspace_id == workspace_id,
                )
            )
            job = db.scalar(select(Job).where(Job.id == job_id, Job.workspace_id == workspace_id))
            if run:
                run.status = "failed"
            if job:
                job.status = "failed"
                job.detail = f"Evaluation failed: {type(exc).__name__}"
            db.commit()
            raise


def run_experiment(
    workspace_id: str,
    experiment_id: str,
    job_id: str,
    revision_id: str,
    payload: dict,
) -> dict[str, str]:
    """Compare named retrieval-depth variants with paired bootstrap intervals."""
    with SessionLocal() as db:
        if db.bind and db.bind.dialect.name == "postgresql":
            db.execute(
                text("SELECT set_config('app.workspace_id', :wid, true)"), {"wid": workspace_id}
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
        experiment.status = "running"
        job.status = "running"
        db.commit()
        try:
            if db.bind and db.bind.dialect.name == "postgresql":
                db.execute(
                    text("SELECT set_config('app.workspace_id', :wid, true)"), {"wid": workspace_id}
                )
            elements = list(
                db.scalars(
                    select(CorpusElement).where(
                        CorpusElement.workspace_id == workspace_id,
                        CorpusElement.embedding_json.is_not(None),
                    )
                )
            )
            dataset = EvaluationDataset(
                dataset_id=f"{revision.dataset_id}:r{revision.revision}",
                sha256=revision.sha256,
                cases=json.loads(revision.cases_json),
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
                answerer = GroundedAnswerGenerator(
                    retriever,
                    OllamaLLMProvider(
                        model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b"),
                        host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                    ),
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
            if db.bind and db.bind.dialect.name == "postgresql":
                db.execute(
                    text("SELECT set_config('app.workspace_id', :wid, true)"), {"wid": workspace_id}
                )
            experiment = db.scalar(
                select(Experiment).where(
                    Experiment.id == experiment_id,
                    Experiment.workspace_id == workspace_id,
                )
            )
            job = db.scalar(select(Job).where(Job.id == job_id, Job.workspace_id == workspace_id))
            if experiment is None or job is None:
                raise LookupError("Experiment state disappeared during execution.")
            experiment.status = report.status
            experiment.report_json = report.model_dump_json()
            job.status = "complete" if report.status == "complete" else report.status
            job.detail = f"{len(report.variants)} variants compared"
            db.commit()
            return {"experiment_id": experiment_id, "status": report.status}
        except Exception:
            db.rollback()
            if db.bind and db.bind.dialect.name == "postgresql":
                db.execute(
                    text("SELECT set_config('app.workspace_id', :wid, true)"), {"wid": workspace_id}
                )
            experiment = db.scalar(
                select(Experiment).where(
                    Experiment.id == experiment_id,
                    Experiment.workspace_id == workspace_id,
                )
            )
            job = db.scalar(select(Job).where(Job.id == job_id, Job.workspace_id == workspace_id))
            if experiment:
                experiment.status = "failed"
            if job:
                job.status = "failed"
                job.detail = "Experiment failed; inspect worker logs for sanitized exception type."
            db.commit()
            raise
