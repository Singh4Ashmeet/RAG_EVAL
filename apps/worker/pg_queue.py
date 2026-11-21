"""PostgreSQL-backed durable job queue for RAG_EVAL.

Provides transactional job claiming, lease fencing, append-only attempt history,
periodic reaping, and bounded execution inside the web application lifespan.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import socket
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import (
    insert,
    select,
    text,
    update,
)
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

UUID_REGEX = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class StaleLeaseError(RuntimeError):
    """Raised when an operation is rejected because a lease expired or was preempted."""


class MalformedPayloadError(ValueError):
    """Raised when a job payload is malformed or invalid."""


# ---------------------------------------------------------------------------
# Versioned Payload Schemas
# ---------------------------------------------------------------------------


class IndexPayloadV1(BaseModel):
    version: Literal[1] = 1
    kind: Literal["index"] = "index"
    document_id: str

    @field_validator("document_id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        if not UUID_REGEX.match(v):
            raise ValueError(f"document_id '{v}' is not a valid UUID")
        return v


class EvaluatePayloadV1(BaseModel):
    version: Literal[1] = 1
    kind: Literal["evaluate"] = "evaluate"
    run_id: str
    revision_id: str
    cutoffs: list[int] = Field(min_length=1)

    @field_validator("run_id", "revision_id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        if not UUID_REGEX.match(v):
            raise ValueError(f"Identifier '{v}' is not a valid UUID")
        return v


class ExperimentPayloadV1(BaseModel):
    version: Literal[1] = 1
    kind: Literal["experiment"] = "experiment"
    experiment_id: str
    revision_id: str
    payload: dict[str, Any]

    @field_validator("experiment_id", "revision_id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        if not UUID_REGEX.match(v):
            raise ValueError(f"Identifier '{v}' is not a valid UUID")
        return v


def validate_payload(kind: str, raw_payload: Any) -> BaseModel:
    """Validate raw payload against the matching versioned schema."""
    if not isinstance(raw_payload, dict):
        raise MalformedPayloadError("Payload must be a JSON object")
    version = raw_payload.get("version")
    if version != 1:
        raise MalformedPayloadError(f"Unsupported payload version: {version}")
    p_kind = raw_payload.get("kind")
    if p_kind != kind:
        raise MalformedPayloadError(f"Payload kind '{p_kind}' does not match job kind '{kind}'")
    try:
        if kind == "index":
            return IndexPayloadV1.model_validate(raw_payload)
        elif kind == "evaluate":
            return EvaluatePayloadV1.model_validate(raw_payload)
        elif kind == "experiment":
            return ExperimentPayloadV1.model_validate(raw_payload)
        else:
            raise MalformedPayloadError(f"Unknown job kind: {kind}")
    except ValidationError as exc:
        raise MalformedPayloadError(f"Payload validation failed: {exc}") from exc


# Database Primitives
def enqueue_job(
    db: Session,
    workspace_id: str,
    kind: str,
    payload: dict[str, Any],
    max_attempts: int = 3,
    detail: str = "",
) -> str:
    """Enqueue a new job with a validated versioned payload."""
    from apps.api.main import Job

    # Validate payload structure before persistence
    validate_payload(kind, payload)

    if db.bind and db.bind.dialect.name == "postgresql":
        # Call SECURITY DEFINER enqueue_job function on PostgreSQL
        try:
            row = (
                db.execute(
                    text(
                        "SELECT public.enqueue_job("
                        ":ws, :kind, CAST(:payload AS jsonb), :max_attempts, :detail"
                        ") AS job_id"
                    ),
                    {
                        "ws": workspace_id,
                        "kind": kind,
                        "payload": json.dumps(payload),
                        "max_attempts": max_attempts,
                        "detail": detail,
                    },
                )
                .mappings()
                .first()
            )
            db.commit()
            if row is not None:
                return str(row["job_id"])
        except Exception as exc:
            db.rollback()
            err_str = str(exc).lower()
            if "does not exist" in err_str or "undefinedfunction" in err_str:
                logger.debug("Falling back to Python enqueue implementation: %s", exc)
            else:
                raise

    # Dialect-agnostic (or fallback) enqueue implementation
    job_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    stmt = insert(Job).values(
        id=job_id,
        workspace_id=workspace_id,
        kind=kind,
        status="queued",
        detail=detail,
        payload_json=payload,
        max_attempts=max_attempts,
        cycle_attempts=0,
        cumulative_attempts=0,
        manual_retry_count=0,
        scheduled_at=now,
        created_at=now,
        updated_at=now,
    )
    db.execute(stmt)
    db.commit()
    return job_id


def claim_next_job(
    db: Session,
    worker_id: str,
    lease_seconds: int = 60,
) -> dict[str, Any] | None:
    """Atomically claim the next eligible queued job using SKIP LOCKED."""
    from apps.api.main import Job, JobAttempt

    if db.bind and db.bind.dialect.name == "postgresql":
        # Call PostgreSQL stored function if available
        try:
            row = (
                db.execute(
                    text("SELECT * FROM public.claim_next_job(:wid, :lease_sec)"),
                    {"wid": worker_id, "lease_sec": lease_seconds},
                )
                .mappings()
                .first()
            )
            db.commit()
            if row:
                return dict(row)
            return None
        except Exception as exc:
            db.rollback()
            # If function doesn't exist yet, fall back to SQL implementation
            logger.debug("Falling back to Python SQL claim implementation: %s", exc)

    # Dialect-agnostic (or fallback) claim implementation
    now = datetime.now(UTC)
    lease_expiry = now + timedelta(seconds=lease_seconds)

    # Row-lock eligible queued job
    query = (
        select(Job)
        .where(
            Job.status == "queued",
            Job.scheduled_at <= now,
        )
        .order_by(Job.scheduled_at.asc(), Job.created_at.asc())
        .limit(1)
    )
    if db.bind and db.bind.dialect.name == "postgresql":
        query = query.with_for_update(skip_locked=True)

    job = db.scalar(query)
    if not job:
        db.commit()
        return None

    # Validate payload
    try:
        validate_payload(job.kind, job.payload_json)
    except Exception as exc:
        # Terminally fail malformed job
        job.status = "failed"
        job.detail = f"Malformed payload: {exc}"
        job.last_error = str(exc)
        job.lease_token = None
        job.lease_expires_at = None
        job.worker_id = None
        job.updated_at = now
        attempt = JobAttempt(
            id=str(uuid.uuid4()),
            job_id=job.id,
            workspace_id=job.workspace_id,
            attempt_number=job.cumulative_attempts + 1,
            lease_generation=job.lease_generation + 1,
            lease_token="invalid",
            worker_id=worker_id,
            started_at=now,
            finished_at=now,
            status="failed",
            error_message=str(exc),
            error_classification="malformed_payload",
        )
        db.add(attempt)
        db.commit()
        return None

    new_token = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    new_gen = job.lease_generation + 1
    new_cycle = job.cycle_attempts + 1
    new_cumulative = job.cumulative_attempts + 1

    job.status = "running"
    job.worker_id = worker_id
    job.lease_token = new_token
    job.lease_generation = new_gen
    job.lease_expires_at = lease_expiry
    job.cycle_attempts = new_cycle
    job.cumulative_attempts = new_cumulative
    job.started_at = job.started_at or now
    job.updated_at = now

    attempt = JobAttempt(
        id=attempt_id,
        job_id=job.id,
        workspace_id=job.workspace_id,
        attempt_number=new_cumulative,
        lease_generation=new_gen,
        lease_token=new_token,
        worker_id=worker_id,
        started_at=now,
        status="running",
    )
    db.add(attempt)
    db.commit()

    return {
        "id": job.id,
        "workspace_id": job.workspace_id,
        "kind": job.kind,
        "payload_json": job.payload_json,
        "lease_token": new_token,
        "lease_generation": new_gen,
        "attempt_id": attempt_id,
        "cycle_attempts": new_cycle,
        "cumulative_attempts": new_cumulative,
    }


def renew_lease(
    db: Session,
    job_id: str,
    lease_token: str,
    lease_generation: int,
    worker_id: str,
    lease_seconds: int = 60,
) -> bool:
    """Renew active lease for the executing worker using database wall-clock time atomically."""
    from apps.api.main import Job

    if db.bind and db.bind.dialect.name == "postgresql":
        # Use clock_timestamp() directly in the UPDATE for atomic lease validation and renewal
        # Use make_interval for proper interval construction
        stmt = (
            update(Job)
            .where(
                Job.id == job_id,
                Job.lease_token == lease_token,
                Job.lease_generation == lease_generation,
                Job.worker_id == worker_id,
                Job.status == "running",
                Job.lease_expires_at > text("clock_timestamp()"),
            )
            .values(
                lease_expires_at=text("clock_timestamp() + make_interval(secs => :lease_sec)"),
                updated_at=text("clock_timestamp()"),
            )
        )
        res = db.execute(stmt, {"lease_sec": lease_seconds})
    else:
        # Fallback for non-PostgreSQL (shouldn't happen in production)
        db_time = datetime.now(UTC)
        new_expiry = db_time + timedelta(seconds=lease_seconds)
        stmt = (
            update(Job)
            .where(
                Job.id == job_id,
                Job.lease_token == lease_token,
                Job.lease_generation == lease_generation,
                Job.worker_id == worker_id,
                Job.status == "running",
                Job.lease_expires_at > db_time,
            )
            .values(
                lease_expires_at=new_expiry,
                updated_at=db_time,
            )
        )
        res = db.execute(stmt)
    db.commit()
    rowcount = getattr(res, "rowcount", 0)
    return rowcount > 0


def complete_job(
    db: Session,
    job_id: str,
    lease_token: str,
    lease_generation: int,
    worker_id: str,
    attempt_id: str,
    detail: str = "",
) -> bool:
    """Finalize job completion atomically within the current transaction."""
    from apps.api.main import Job, JobAttempt

    # Row-level lock the job to prevent concurrent modifications
    job = db.scalar(
        select(Job)
        .where(
            Job.id == job_id,
            Job.lease_token == lease_token,
            Job.lease_generation == lease_generation,
            Job.worker_id == worker_id,
            Job.status == "running",
        )
        .with_for_update()
    )
    if not job:
        db.commit()
        return False

    # Get database wall-clock time AFTER acquiring the lock
    if db.bind and db.bind.dialect.name == "postgresql":
        db_time = db.scalar(select(text("clock_timestamp()")))
    else:
        db_time = datetime.now(UTC)

    # Validate lease hasn't expired using wall-clock time
    if job.lease_expires_at is not None and job.lease_expires_at <= db_time:
        db.commit()
        return False

    job.status = "complete"
    job.detail = detail
    job.lease_token = None
    job.lease_expires_at = None
    job.worker_id = None
    job.completed_at = db_time
    job.updated_at = db_time

    # Update JobAttempt with full ownership validation including job_id and workspace_id
    res = db.execute(
        update(JobAttempt)
        .where(
            JobAttempt.id == attempt_id,
            JobAttempt.job_id == job_id,
            JobAttempt.workspace_id == job.workspace_id,
            JobAttempt.lease_generation == lease_generation,
            JobAttempt.lease_token == lease_token,
            JobAttempt.worker_id == worker_id,
            JobAttempt.status == "running",
        )
        .values(
            status="complete",
            finished_at=db_time,
        )
    )

    # Require exactly one affected row; otherwise rollback and fail
    if getattr(res, "rowcount", 0) != 1:
        db.rollback()
        return False

    db.commit()
    return True


def fail_job(
    db: Session,
    job_id: str,
    lease_token: str,
    lease_generation: int,
    worker_id: str,
    attempt_id: str,
    error: str,
    is_retryable: bool = False,
    backoff_base: int = 10,
    backoff_max: int = 300,
    error_classification: str = "internal_error",
) -> bool:
    """Record task failure with backoff retry or terminal failure.

    Uses row-level locking and database wall-clock time lease validation to ensure
    concurrency safety and consistency with the reaper behavior.
    """
    from apps.api.main import Document, EvaluationRun, Experiment, Job, JobAttempt

    # Row-level lock the job to prevent concurrent modifications
    job = db.scalar(
        select(Job)
        .where(
            Job.id == job_id,
            Job.lease_token == lease_token,
            Job.lease_generation == lease_generation,
            Job.worker_id == worker_id,
            Job.status == "running",
        )
        .with_for_update()
    )
    if not job:
        db.commit()
        return False

    # Get database wall-clock time AFTER acquiring the lock
    if db.bind and db.bind.dialect.name == "postgresql":
        db_time = db.scalar(select(text("clock_timestamp()")))
    else:
        db_time = datetime.now(UTC)

    # Validate lease hasn't expired using wall-clock time
    if job.lease_expires_at is not None and job.lease_expires_at <= db_time:
        db.commit()
        return False

    should_retry = is_retryable and (job.cycle_attempts < job.max_attempts)
    clean_error = error[:2048]

    # Initialize domain_sync_error before the branch so it's always defined
    domain_sync_error = None

    # Synchronize domain entities (matching reaper behavior) BEFORE modifying Job fields.
    # Use SQLAlchemy's nested transaction (savepoint) so domain-sync failures don't
    # abort the outer transaction.
    payload = job.payload_json or {}
    try:
        with db.begin_nested():
            if job.kind == "index":
                doc_id = payload.get("document_id")
                if doc_id and UUID_REGEX.match(doc_id):
                    db.execute(
                        update(Document)
                        .where(
                            Document.id == doc_id,
                            Document.workspace_id == job.workspace_id,
                            Document.status != "indexed",
                        )
                        .values(status="failed")
                    )
            elif job.kind == "evaluate":
                run_id = payload.get("run_id")
                if run_id and UUID_REGEX.match(run_id):
                    db.execute(
                        update(EvaluationRun)
                        .where(
                            EvaluationRun.id == run_id,
                            EvaluationRun.workspace_id == job.workspace_id,
                            EvaluationRun.status != "complete",
                        )
                        .values(status="failed")
                    )
            elif job.kind == "experiment":
                exp_id = payload.get("experiment_id")
                if exp_id and UUID_REGEX.match(exp_id):
                    db.execute(
                        update(Experiment)
                        .where(
                            Experiment.id == exp_id,
                            Experiment.workspace_id == job.workspace_id,
                            Experiment.status != "complete",
                        )
                        .values(status="failed")
                    )
    except Exception as exc:
        # Record domain sync error for later, preserving the original task error
        domain_sync_error = exc
        logger.error("Domain synchronization failed for job %s: %s", job_id, exc)

    # Now update Job fields based on retry/terminal path
    if should_retry:
        delay = min(backoff_max, backoff_base * (2 ** max(0, job.cycle_attempts - 1)))
        jitter = delay * (0.8 + 0.4 * random.random())
        scheduled = db_time + timedelta(seconds=jitter)
        job.status = "queued"
        job.scheduled_at = scheduled
        job.detail = f"Retry scheduled in {int(jitter)}s: {clean_error}"
    else:
        job.status = "failed"
        # Record both task error and domain sync error (if any)
        if domain_sync_error is not None:
            job.last_error = f"Domain sync error: {domain_sync_error}; task error: {clean_error}"
            job.detail = f"Failed: {clean_error}; domain sync failed: {domain_sync_error}"
        else:
            job.last_error = clean_error
            job.detail = f"Failed: {clean_error}"

    # Common lease field cleanup
    job.lease_token = None
    job.lease_expires_at = None
    job.worker_id = None
    job.updated_at = db_time

    # Update JobAttempt with full ownership validation including job_id and workspace_id
    res = db.execute(
        update(JobAttempt)
        .where(
            JobAttempt.id == attempt_id,
            JobAttempt.job_id == job_id,
            JobAttempt.workspace_id == job.workspace_id,
            JobAttempt.lease_generation == lease_generation,
            JobAttempt.lease_token == lease_token,
            JobAttempt.worker_id == worker_id,
            JobAttempt.status == "running",
        )
        .values(
            status="failed" if not should_retry else "timeout",
            finished_at=db_time,
            error_message=clean_error,
            error_classification=error_classification,
        )
    )

    # Require exactly one affected row; otherwise rollback and fail
    if getattr(res, "rowcount", 0) != 1:
        db.rollback()
        return False

    db.commit()
    return True


def reap_expired_jobs(
    db: Session,
    base_delay: int = 10,
    max_delay: int = 300,
) -> list[dict[str, Any]]:
    """Reap jobs whose leases have expired and requeue or terminally fail them."""
    if db.bind and db.bind.dialect.name == "postgresql":
        try:
            rows = (
                db.execute(
                    text("SELECT * FROM public.reap_expired_jobs(:base, :max)"),
                    {"base": base_delay, "max": max_delay},
                )
                .mappings()
                .all()
            )
            db.commit()
            return [dict(r) for r in rows]
        except Exception as exc:
            db.rollback()
            logger.debug("Falling back to Python SQL reap implementation: %s", exc)

    now = datetime.now(UTC)
    from apps.api.main import Document, EvaluationRun, Experiment, Job, JobAttempt

    query = select(Job).where(
        Job.status == "running",
        Job.lease_expires_at <= now,
    )
    if db.bind and db.bind.dialect.name == "postgresql":
        query = query.with_for_update(skip_locked=True)

    expired_jobs = list(db.scalars(query))
    reaped = []

    for job in expired_jobs:
        if job.cycle_attempts < job.max_attempts:
            # Requeue with backoff + jitter
            delay = min(max_delay, base_delay * (2 ** max(0, job.cycle_attempts - 1)))
            jitter = delay * (0.8 + 0.4 * random.random())
            job.status = "queued"
            job.scheduled_at = now + timedelta(seconds=jitter)
            job.detail = "Recovered after worker timeout; queued for retry"
            final_status = "queued"

            db.execute(
                update(JobAttempt)
                .where(
                    JobAttempt.job_id == job.id,
                    JobAttempt.lease_generation == job.lease_generation,
                    JobAttempt.status == "running",
                )
                .values(
                    status="timeout",
                    finished_at=now,
                    error_message="Lease expired without renewal",
                    error_classification="lease_timeout",
                )
            )
        else:
            # Terminally fail
            job.status = "failed"
            job.detail = "Execution timed out; maximum retry attempts reached"
            job.last_error = "Lease expired without renewal. Max retry attempts exhausted."
            final_status = "failed"

            db.execute(
                update(JobAttempt)
                .where(
                    JobAttempt.job_id == job.id,
                    JobAttempt.lease_generation == job.lease_generation,
                    JobAttempt.status == "running",
                )
                .values(
                    status="failed",
                    finished_at=now,
                    error_message="Max retry attempts reached after timeout",
                    error_classification="max_attempts_exceeded",
                )
            )

            # Synchronize domain entities
            payload = job.payload_json or {}
            if job.kind == "index":
                doc_id = payload.get("document_id")
                if doc_id:
                    db.execute(
                        update(Document)
                        .where(
                            Document.id == doc_id,
                            Document.workspace_id == job.workspace_id,
                            Document.status != "indexed",
                        )
                        .values(status="failed")
                    )
            elif job.kind == "evaluate":
                run_id = payload.get("run_id")
                if run_id:
                    db.execute(
                        update(EvaluationRun)
                        .where(
                            EvaluationRun.id == run_id,
                            EvaluationRun.workspace_id == job.workspace_id,
                            EvaluationRun.status != "complete",
                        )
                        .values(status="failed")
                    )
            elif job.kind == "experiment":
                exp_id = payload.get("experiment_id")
                if exp_id:
                    db.execute(
                        update(Experiment)
                        .where(
                            Experiment.id == exp_id,
                            Experiment.workspace_id == job.workspace_id,
                            Experiment.status != "complete",
                        )
                        .values(status="failed")
                    )

        job.lease_token = None
        job.lease_expires_at = None
        job.worker_id = None
        job.updated_at = now
        reaped.append(
            {
                "id": job.id,
                "workspace_id": job.workspace_id,
                "kind": job.kind,
                "status": final_status,
            }
        )

    db.commit()
    return reaped


def manual_retry_job(
    db: Session,
    job_id: str,
    workspace_id: str,
) -> dict[str, Any] | None:
    """Manually retry a failed job, preserving cumulative attempt history."""
    if db.bind and db.bind.dialect.name == "postgresql":
        try:
            row = (
                db.execute(
                    text("SELECT * FROM public.manual_retry_job(:job_id, :workspace_id)"),
                    {"job_id": job_id, "workspace_id": workspace_id},
                )
                .mappings()
                .first()
            )
            db.commit()
            if row:
                return dict(row)
            return None
        except Exception as exc:
            db.rollback()
            err_str = str(exc).lower()
            if "does not exist" in err_str or "undefinedfunction" in err_str:
                logger.debug("Falling back to Python manual_retry_job: %s", exc)
            else:
                raise

    from apps.api.main import Document, EvaluationRun, Experiment, Job

    now = datetime.now(UTC)
    job = db.scalar(
        select(Job)
        .where(
            Job.id == job_id,
            Job.workspace_id == workspace_id,
            Job.status == "failed",
        )
        .with_for_update()
        if db.bind and db.bind.dialect.name == "postgresql"
        else select(Job).where(
            Job.id == job_id,
            Job.workspace_id == workspace_id,
            Job.status == "failed",
        )
    )
    if not job:
        db.commit()
        return None

    job.status = "queued"
    job.detail = "Manually retried by operator"
    job.cycle_attempts = 0
    job.manual_retry_count += 1
    job.lease_token = None
    job.lease_expires_at = None
    job.worker_id = None
    job.scheduled_at = now
    job.last_error = None
    job.updated_at = now

    # Reset corresponding domain entity status
    payload = job.payload_json or {}
    if job.kind == "index":
        doc_id = payload.get("document_id")
        if doc_id:
            db.execute(
                update(Document)
                .where(Document.id == doc_id, Document.workspace_id == workspace_id)
                .values(status="queued")
            )
    elif job.kind == "evaluate":
        run_id = payload.get("run_id")
        if run_id:
            db.execute(
                update(EvaluationRun)
                .where(EvaluationRun.id == run_id, EvaluationRun.workspace_id == workspace_id)
                .values(status="queued")
            )
    elif job.kind == "experiment":
        exp_id = payload.get("experiment_id")
        if exp_id:
            db.execute(
                update(Experiment)
                .where(Experiment.id == exp_id, Experiment.workspace_id == workspace_id)
                .values(status="queued")
            )

    db.commit()
    return {
        "id": job.id,
        "status": "queued",
        "cycle_attempts": 0,
        "manual_retry_count": job.manual_retry_count,
    }


# ---------------------------------------------------------------------------
# In-Process Job Processor Loop
# ---------------------------------------------------------------------------


class JobProcessor:
    """Manages the in-process worker loop and reaper inside FastAPI lifespan."""

    def __init__(
        self,
        session_factory: sessionmaker,
        worker_id: str | None = None,
        poll_interval: float = 2.0,
        reaper_interval: float = 30.0,
        lease_seconds: int = 60,
        heartbeat_interval: float = 20.0,
    ):
        self.session_factory = session_factory
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.poll_interval = poll_interval
        self.reaper_interval = reaper_interval
        self.lease_seconds = lease_seconds
        self.heartbeat_interval = heartbeat_interval
        self.shutdown_event = asyncio.Event()
        self.is_running = False
        self._main_task: asyncio.Task | None = None

    def start(self) -> asyncio.Task:
        """Start the worker loop background task."""
        if self.is_running and self._main_task is not None:
            return self._main_task
        self.is_running = True
        self.shutdown_event.clear()
        self._main_task = asyncio.create_task(self._run_loop())
        return self._main_task

    async def stop(self, timeout: float = 10.0) -> None:  # noqa: ASYNC109
        """Signal worker loop to stop and wait for active task grace period."""
        if not self.is_running:
            return
        logger.info("Stopping job processor worker %s...", self.worker_id)
        self.shutdown_event.set()
        self.is_running = False
        if self._main_task:
            try:
                await asyncio.wait_for(asyncio.shield(self._main_task), timeout=timeout)
            except TimeoutError:
                logger.warning(
                    "Worker %s active task exceeded shutdown timeout of %.1fs",
                    self.worker_id,
                    timeout,
                )
            except Exception as exc:
                logger.error("Error awaiting worker task termination: %s", exc)

    async def _run_loop(self) -> None:
        """Main loop: periodically reaps expired leases and claims eligible jobs."""
        logger.info("Job processor started with worker_id=%s", self.worker_id)
        last_reap = 0.0

        while not self.shutdown_event.is_set():
            loop_now = asyncio.get_event_loop().time()
            if loop_now - last_reap >= self.reaper_interval:
                try:
                    await asyncio.to_thread(self._reap)
                except Exception as exc:
                    logger.error("Error during reaper execution: %s", exc)
                last_reap = loop_now

            job_claim = None
            try:
                job_claim = await asyncio.to_thread(self._claim)
            except Exception as exc:
                logger.error("Error claiming next job: %s", exc)

            if job_claim:
                try:
                    await self._process_job(job_claim)
                except Exception as exc:
                    logger.error("Unhandled error processing job %s: %s", job_claim.get("id"), exc)
            else:
                try:
                    await asyncio.wait_for(self.shutdown_event.wait(), timeout=self.poll_interval)
                except TimeoutError:
                    pass

        logger.info("Job processor loop terminated.")

    def _claim(self) -> dict[str, Any] | None:
        with self.session_factory() as db:
            return claim_next_job(db, self.worker_id, self.lease_seconds)

    def _reap(self) -> list[dict[str, Any]]:
        with self.session_factory() as db:
            return reap_expired_jobs(db)

    async def _process_job(self, claim: dict[str, Any]) -> None:
        """Execute claimed task with background heartbeat."""
        job_id = claim["id"]
        token = claim["lease_token"]
        gen = claim["lease_generation"]
        attempt_id = claim["attempt_id"]
        kind = claim["kind"]
        workspace_id = claim["workspace_id"]
        payload = claim["payload_json"]

        stop_heartbeat = asyncio.Event()
        heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(job_id, token, gen, stop_heartbeat)
        )

        try:
            from apps.worker.tasks import (
                evaluate_dataset,
                index_document,
                run_experiment,
            )

            if kind == "index":
                await asyncio.to_thread(
                    index_document,
                    workspace_id=workspace_id,
                    document_id=payload["document_id"],
                    job_id=job_id,
                    lease_token=token,
                    lease_generation=gen,
                    worker_id=self.worker_id,
                    attempt_id=attempt_id,
                    session_factory=self.session_factory,
                )
            elif kind == "evaluate":
                await asyncio.to_thread(
                    evaluate_dataset,
                    workspace_id=workspace_id,
                    run_id=payload["run_id"],
                    job_id=job_id,
                    revision_id=payload["revision_id"],
                    cutoffs=payload["cutoffs"],
                    lease_token=token,
                    lease_generation=gen,
                    worker_id=self.worker_id,
                    attempt_id=attempt_id,
                    session_factory=self.session_factory,
                )
            elif kind == "experiment":
                await asyncio.to_thread(
                    run_experiment,
                    workspace_id=workspace_id,
                    experiment_id=payload["experiment_id"],
                    job_id=job_id,
                    revision_id=payload["revision_id"],
                    payload=payload["payload"],
                    lease_token=token,
                    lease_generation=gen,
                    worker_id=self.worker_id,
                    attempt_id=attempt_id,
                    session_factory=self.session_factory,
                )
        except StaleLeaseError:
            logger.warning("Job %s aborted because lease was lost.", job_id)
        except Exception as exc:
            logger.error("Job %s execution failed: %s", job_id, exc)
            with self.session_factory() as db:
                fail_job(
                    db,
                    job_id=job_id,
                    lease_token=token,
                    lease_generation=gen,
                    worker_id=self.worker_id,
                    attempt_id=attempt_id,
                    error=str(exc),
                    is_retryable=False,
                    error_classification=type(exc).__name__,
                )
        finally:
            stop_heartbeat.set()
            try:
                await heartbeat_task
            except Exception:
                pass

    async def _heartbeat_loop(
        self,
        job_id: str,
        token: str,
        gen: int,
        stop_event: asyncio.Event,
    ) -> None:
        """Periodically renew lease in background while task runs."""
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.heartbeat_interval)
                break
            except TimeoutError:
                pass

            if stop_event.is_set():
                break

            try:
                with self.session_factory() as db:
                    renewed = renew_lease(
                        db,
                        job_id=job_id,
                        lease_token=token,
                        lease_generation=gen,
                        worker_id=self.worker_id,
                        lease_seconds=self.lease_seconds,
                    )
                if not renewed:
                    logger.warning("Heartbeat failed for job %s: lease lost or preempted.", job_id)
                    break
            except Exception as exc:
                logger.error("Error during heartbeat renewal for job %s: %s", job_id, exc)
