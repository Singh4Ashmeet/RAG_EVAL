"""add durable job queue columns, job_attempts, and queue functions

Revision ID: 20261109_0003
Revises: 20261101_0002
Create Date: 2026-11-09
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20261109_0003"
down_revision = "20261101_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_pg = bind.dialect.name == "postgresql"

    # 0. Prerequisite Role Verification (PostgreSQL only)
    if is_pg:
        op.execute(
            sa.text("""
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_api_queue_owner') OR
       NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_worker_queue_owner') OR
       NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_api') OR
       NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_worker') THEN
        RAISE EXCEPTION 'Migration prerequisite failure: Required roles (rag_api_queue_owner, rag_worker_queue_owner, rag_api, rag_worker) do not exist. Execute deploy/sql/01_worker_roles_and_privileges.sql first.';
    END IF;
END
$$;
""")
        )

    # 1. Add new columns to jobs table
    json_col = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")
    op.add_column("jobs", sa.Column("payload_json", json_col, nullable=True))
    op.add_column("jobs", sa.Column("worker_id", sa.String(64), nullable=True))
    op.add_column("jobs", sa.Column("lease_token", sa.String(36), nullable=True))
    op.add_column(
        "jobs", sa.Column("lease_generation", sa.Integer(), nullable=False, server_default="0")
    )
    op.add_column("jobs", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "jobs", sa.Column("cycle_attempts", sa.Integer(), nullable=False, server_default="0")
    )
    op.add_column(
        "jobs", sa.Column("cumulative_attempts", sa.Integer(), nullable=False, server_default="0")
    )
    op.add_column(
        "jobs", sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3")
    )
    op.add_column(
        "jobs", sa.Column("manual_retry_count", sa.Integer(), nullable=False, server_default="0")
    )
    op.add_column(
        "jobs",
        sa.Column(
            "scheduled_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.add_column("jobs", sa.Column("last_error", sa.Text(), nullable=True))
    op.add_column("jobs", sa.Column("started_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "jobs",
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.add_column("jobs", sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True))

    # 2. Create job_attempts table
    op.create_table(
        "job_attempts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "job_id", sa.String(36), sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "workspace_id",
            sa.String(36),
            sa.ForeignKey("accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("lease_generation", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("lease_token", sa.String(36), nullable=False),
        sa.Column("worker_id", sa.String(64), nullable=False),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(24), nullable=False, server_default="running"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("error_classification", sa.String(64), nullable=True),
    )
    op.create_index("ix_job_attempts_job_id", "job_attempts", ["job_id"])
    op.create_index("ix_job_attempts_workspace_id", "job_attempts", ["workspace_id"])
    if is_pg:
        op.create_index(
            "uq_job_attempts_active_job",
            "job_attempts",
            ["job_id"],
            unique=True,
            postgresql_where=sa.text("status = 'running'"),
        )

        # 2b. Establish Table Privileges on queue tables immediately upon creation
        op.execute(
            sa.text("""
-- Table privileges for rag_api: SELECT only on jobs and job_attempts (strictly no direct DML)
GRANT SELECT ON public.jobs TO rag_api;
REVOKE INSERT, UPDATE, DELETE ON public.jobs FROM rag_api;

GRANT SELECT ON public.job_attempts TO rag_api;
REVOKE INSERT, UPDATE, DELETE ON public.job_attempts FROM rag_api;

-- Table privileges for rag_worker: SELECT, UPDATE on jobs; SELECT, INSERT, UPDATE on job_attempts
GRANT SELECT, UPDATE ON public.jobs TO rag_worker;
REVOKE INSERT, DELETE ON public.jobs FROM rag_worker;

GRANT SELECT, INSERT, UPDATE ON public.job_attempts TO rag_worker;
REVOKE DELETE ON public.job_attempts FROM rag_worker;

-- Table privileges for rag_api_queue_owner (API queue functions: enqueue_job, manual_retry_job)
-- Note: enqueue_job and manual_retry_job only touch jobs, documents, evaluation_runs, and experiments.
-- Least privilege: rag_api_queue_owner receives strictly NO privileges on job_attempts.
GRANT SELECT, INSERT, UPDATE ON public.jobs TO rag_api_queue_owner;
REVOKE ALL ON public.job_attempts FROM rag_api_queue_owner;
GRANT SELECT, UPDATE ON public.documents TO rag_api_queue_owner;
GRANT SELECT, UPDATE ON public.evaluation_runs TO rag_api_queue_owner;
GRANT SELECT, UPDATE ON public.experiments TO rag_api_queue_owner;

-- Table privileges for rag_worker_queue_owner (Worker queue functions: claim_next_job, reap_expired_jobs)
GRANT SELECT, UPDATE ON public.jobs TO rag_worker_queue_owner;
GRANT SELECT, INSERT, UPDATE ON public.job_attempts TO rag_worker_queue_owner;
GRANT SELECT, UPDATE ON public.documents TO rag_worker_queue_owner;
GRANT SELECT, UPDATE ON public.evaluation_runs TO rag_worker_queue_owner;
GRANT SELECT, UPDATE ON public.experiments TO rag_worker_queue_owner;
""")
        )

    # 3. Create indexes on jobs table
    op.create_index("ix_jobs_queue_claim", "jobs", ["status", "scheduled_at", "lease_expires_at"])
    op.create_index("ix_jobs_lease_token", "jobs", ["lease_token"])

    # 4. Migrate pre-existing jobs
    if is_pg:
        # Terminate legacy queued/running jobs whose RQ payload was not persisted in PostgreSQL
        op.execute(
            sa.text("""
            UPDATE public.jobs
            SET status = 'failed',
                detail = 'Terminated during durable queue migration: legacy RQ payload unavailable',
                last_error = 'Migration 20261109_0003: Job predates durable PostgreSQL queue payload persistence.',
                updated_at = clock_timestamp()
            WHERE status IN ('queued', 'running')
              AND payload_json IS NULL;
        """)
        )
        # Provide stub payload for historical completed/failed jobs
        op.execute(
            sa.text("""
            UPDATE public.jobs
            SET payload_json = jsonb_build_object('version', 1, 'kind', kind, 'legacy', true)
            WHERE payload_json IS NULL;
        """)
        )
    else:
        op.execute(
            sa.text("""
            UPDATE jobs
            SET status = 'failed',
                detail = 'Terminated during durable queue migration: legacy RQ payload unavailable',
                last_error = 'Migration 20261109_0003: Job predates durable PostgreSQL queue payload persistence.'
            WHERE status IN ('queued', 'running')
              AND payload_json IS NULL;
        """)
        )
        op.execute(
            sa.text("""
            UPDATE jobs
            SET payload_json = '{"version": 1, "legacy": true}'
            WHERE payload_json IS NULL;
        """)
        )

    # 5. Row-Level Security for job_attempts (PostgreSQL only)
    if is_pg:
        op.execute(sa.text('ALTER TABLE "job_attempts" ENABLE ROW LEVEL SECURITY'))
        op.execute(sa.text('ALTER TABLE "job_attempts" FORCE ROW LEVEL SECURITY'))
        op.execute(
            sa.text(
                'CREATE POLICY workspace_isolation ON "job_attempts" '
                "USING (workspace_id = current_setting('app.workspace_id', true)) "
                "WITH CHECK (workspace_id = current_setting('app.workspace_id', true))"
            )
        )

        # 6. Deploy PostgreSQL stored functions directly (self-contained DDL)
        op.execute(
            sa.text("""
CREATE OR REPLACE FUNCTION public.enqueue_job(
    p_workspace_id TEXT,
    p_kind TEXT,
    p_payload JSONB,
    p_max_attempts INT DEFAULT 3,
    p_detail TEXT DEFAULT ''
)
RETURNS TEXT
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    v_caller_ws TEXT;
    v_job_id TEXT;
    v_now TIMESTAMPTZ := clock_timestamp();
BEGIN
    v_caller_ws := current_setting('app.workspace_id', true);
    IF v_caller_ws IS NULL OR v_caller_ws = '' THEN
        RAISE EXCEPTION 'Workspace context not set (app.workspace_id is missing)';
    END IF;
    IF v_caller_ws != p_workspace_id THEN
        RAISE EXCEPTION 'Workspace authorization failed: context=% target=%', v_caller_ws, p_workspace_id;
    END IF;

    IF p_payload IS NULL THEN
        RAISE EXCEPTION 'Payload must not be NULL';
    END IF;
    IF (p_payload->>'version')::INT IS DISTINCT FROM 1 THEN
        RAISE EXCEPTION 'Unsupported payload version: %', COALESCE(p_payload->>'version', 'null');
    END IF;
    IF p_payload->>'kind' IS DISTINCT FROM p_kind THEN
        RAISE EXCEPTION 'Payload kind mismatch: payload=% job=%', COALESCE(p_payload->>'kind', 'null'), p_kind;
    END IF;

    v_job_id := gen_random_uuid()::TEXT;
    INSERT INTO public.jobs (
        id, workspace_id, kind, status, payload_json, detail,
        max_attempts, cycle_attempts, cumulative_attempts, manual_retry_count,
        lease_generation, lease_token, worker_id, lease_expires_at,
        scheduled_at, created_at, updated_at
    ) VALUES (
        v_job_id, p_workspace_id, p_kind, 'queued', p_payload, p_detail,
        p_max_attempts, 0, 0, 0,
        0, NULL, NULL, NULL,
        v_now, v_now, v_now
    );

    RETURN v_job_id;
END;
$$;
""")
        )

        op.execute(
            sa.text("""
CREATE OR REPLACE FUNCTION public.claim_next_job(
    p_worker_id TEXT,
    p_lease_seconds INT DEFAULT 60
)
RETURNS TABLE (
    id VARCHAR(36),
    workspace_id VARCHAR(36),
    kind VARCHAR(32),
    payload_json JSONB,
    lease_token VARCHAR(36),
    lease_generation INT,
    attempt_id VARCHAR(36),
    cycle_attempts INT,
    cumulative_attempts INT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    v_now TIMESTAMPTZ := clock_timestamp();
    v_lease_expiry TIMESTAMPTZ := v_now + (p_lease_seconds || ' seconds')::INTERVAL;
    v_new_token VARCHAR(36);
    v_attempt_id VARCHAR(36);
    v_target_id VARCHAR(36);
    v_is_valid BOOLEAN := TRUE;
    v_error_msg TEXT := NULL;
    r RECORD;
BEGIN
    IF p_worker_id IS NULL OR length(p_worker_id) = 0 THEN
        RAISE EXCEPTION 'Worker ID must not be empty';
    END IF;
    IF p_lease_seconds < 10 OR p_lease_seconds > 600 THEN
        RAISE EXCEPTION 'Lease duration must be between 10 and 600 seconds';
    END IF;

    LOOP
        SELECT j.id, j.workspace_id, j.kind, j.payload_json, j.lease_generation, j.cycle_attempts, j.cumulative_attempts
        INTO r
        FROM public.jobs j
        WHERE j.status = 'queued'
          AND j.scheduled_at <= v_now
        ORDER BY j.scheduled_at ASC, j.created_at ASC
        LIMIT 1
        FOR UPDATE SKIP LOCKED;

        IF NOT FOUND THEN
            RETURN;
        END IF;

        v_is_valid := TRUE;
        v_error_msg := NULL;

        IF r.payload_json IS NULL THEN
            v_is_valid := FALSE;
            v_error_msg := 'Missing required payload_json';
        ELSIF (r.payload_json->>'version')::INT != 1 THEN
            v_is_valid := FALSE;
            v_error_msg := 'Unsupported payload version: ' || COALESCE(r.payload_json->>'version', 'null');
        ELSIF r.payload_json->>'kind' != r.kind THEN
            v_is_valid := FALSE;
            v_error_msg := 'Payload kind mismatch: payload has ' || COALESCE(r.payload_json->>'kind', 'null') || ' but job has ' || r.kind;
        ELSE
            IF r.kind = 'index' THEN
                v_target_id := r.payload_json->>'document_id';
                IF v_target_id IS NULL OR NOT (v_target_id ~ '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$') THEN
                    v_is_valid := FALSE;
                    v_error_msg := 'Invalid or missing document_id in index payload';
                ELSIF NOT EXISTS (SELECT 1 FROM public.documents d WHERE d.id = v_target_id AND d.workspace_id = r.workspace_id) THEN
                    v_is_valid := FALSE;
                    v_error_msg := 'Referenced document does not exist in workspace';
                END IF;
            ELSIF r.kind = 'evaluate' THEN
                v_target_id := r.payload_json->>'run_id';
                IF v_target_id IS NULL OR NOT (v_target_id ~ '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$') THEN
                    v_is_valid := FALSE;
                    v_error_msg := 'Invalid or missing run_id in evaluate payload';
                ELSIF NOT EXISTS (SELECT 1 FROM public.evaluation_runs er WHERE er.id = v_target_id AND er.workspace_id = r.workspace_id) THEN
                    v_is_valid := FALSE;
                    v_error_msg := 'Referenced evaluation_run does not exist in workspace';
                END IF;
            ELSIF r.kind = 'experiment' THEN
                v_target_id := r.payload_json->>'experiment_id';
                IF v_target_id IS NULL OR NOT (v_target_id ~ '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$') THEN
                    v_is_valid := FALSE;
                    v_error_msg := 'Invalid or missing experiment_id in experiment payload';
                ELSIF NOT EXISTS (SELECT 1 FROM public.experiments ex WHERE ex.id = v_target_id AND ex.workspace_id = r.workspace_id) THEN
                    v_is_valid := FALSE;
                    v_error_msg := 'Referenced experiment does not exist in workspace';
                END IF;
            END IF;
        END IF;

        IF v_is_valid THEN
            v_new_token := gen_random_uuid()::TEXT;
            v_attempt_id := gen_random_uuid()::TEXT;

            UPDATE public.jobs j
            SET status = 'running',
                worker_id = p_worker_id,
                lease_token = v_new_token,
                lease_generation = j.lease_generation + 1,
                lease_expires_at = v_lease_expiry,
                cycle_attempts = j.cycle_attempts + 1,
                cumulative_attempts = j.cumulative_attempts + 1,
                started_at = COALESCE(j.started_at, v_now),
                updated_at = v_now
            WHERE j.id = r.id;

            INSERT INTO public.job_attempts (
                id, job_id, workspace_id, attempt_number, lease_generation, lease_token, worker_id, started_at, status
            ) VALUES (
                v_attempt_id, r.id, r.workspace_id, r.cumulative_attempts + 1, r.lease_generation + 1, v_new_token, p_worker_id, v_now, 'running'
            );

            RETURN QUERY SELECT
                r.id,
                r.workspace_id,
                r.kind,
                r.payload_json,
                v_new_token,
                r.lease_generation + 1,
                v_attempt_id,
                r.cycle_attempts + 1,
                r.cumulative_attempts + 1;
            RETURN;
        ELSE
            UPDATE public.jobs j
            SET status = 'failed',
                detail = 'Malformed payload: ' || v_error_msg,
                last_error = v_error_msg,
                lease_token = NULL,
                lease_expires_at = NULL,
                worker_id = NULL,
                updated_at = v_now
            WHERE j.id = r.id;

            INSERT INTO public.job_attempts (
                id, job_id, workspace_id, attempt_number, lease_generation, lease_token, worker_id, started_at, finished_at, status, error_message, error_classification
            ) VALUES (
                gen_random_uuid()::TEXT, r.id, r.workspace_id, r.cumulative_attempts + 1, r.lease_generation + 1, 'invalid', p_worker_id, v_now, v_now, 'failed', v_error_msg, 'malformed_payload'
            );
        END IF;
    END LOOP;
END;
$$;
""")
        )

        op.execute(
            sa.text("""
CREATE OR REPLACE FUNCTION public.reap_expired_jobs(
    p_base_delay_seconds INT DEFAULT 10,
    p_max_delay_seconds INT DEFAULT 300
)
RETURNS TABLE (
    reaped_id VARCHAR(36),
    reaped_workspace_id VARCHAR(36),
    reaped_kind VARCHAR(32),
    final_status VARCHAR(24)
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    v_now TIMESTAMPTZ := clock_timestamp();
    r RECORD;
    v_target_id VARCHAR(36);
BEGIN
    FOR r IN
        SELECT j.id, j.cycle_attempts, j.workspace_id, j.kind, j.lease_generation
        FROM public.jobs j
        WHERE j.status = 'running'
          AND j.lease_expires_at <= v_now
          AND j.cycle_attempts < j.max_attempts
        ORDER BY j.lease_expires_at ASC
        FOR UPDATE SKIP LOCKED
    LOOP
        UPDATE public.jobs
        SET status = 'queued',
            detail = 'Recovered after worker timeout; queued for retry',
            lease_token = NULL,
            lease_expires_at = NULL,
            worker_id = NULL,
            scheduled_at = v_now + (
                LEAST(p_max_delay_seconds, p_base_delay_seconds * POWER(2, GREATEST(0, r.cycle_attempts - 1)))
                * (0.8 + 0.4 * random())
                || ' seconds'
            )::INTERVAL,
            updated_at = v_now
        WHERE id = r.id;

        UPDATE public.job_attempts
        SET status = 'timeout',
            finished_at = v_now,
            error_message = 'Lease expired without renewal; requeued by reaper',
            error_classification = 'lease_timeout'
        WHERE job_id = r.id AND lease_generation = r.lease_generation AND status = 'running';

        reaped_id := r.id;
        reaped_workspace_id := r.workspace_id;
        reaped_kind := r.kind;
        final_status := 'queued';
        RETURN NEXT;
    END LOOP;

    FOR r IN
        SELECT j.id, j.workspace_id, j.kind, j.payload_json, j.lease_generation
        FROM public.jobs j
        WHERE j.status = 'running'
          AND j.lease_expires_at <= v_now
          AND j.cycle_attempts >= j.max_attempts
        ORDER BY j.lease_expires_at ASC
        FOR UPDATE SKIP LOCKED
    LOOP
        UPDATE public.jobs
        SET status = 'failed',
            detail = 'Execution timed out; maximum retry attempts reached',
            last_error = 'Lease expired without renewal. Max retry attempts exhausted.',
            lease_token = NULL,
            lease_expires_at = NULL,
            worker_id = NULL,
            updated_at = v_now
        WHERE id = r.id;

        UPDATE public.job_attempts
        SET status = 'failed',
            finished_at = v_now,
            error_message = 'Max retry attempts reached after timeout',
            error_classification = 'max_attempts_exceeded'
        WHERE job_id = r.id AND lease_generation = r.lease_generation AND status = 'running';

        IF r.payload_json IS NOT NULL THEN
            BEGIN
                IF r.kind = 'index' THEN
                    v_target_id := r.payload_json->>'document_id';
                    IF v_target_id IS NOT NULL AND v_target_id ~ '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$' THEN
                        UPDATE public.documents
                        SET status = 'failed'
                        WHERE id = v_target_id
                          AND workspace_id = r.workspace_id
                          AND status != 'indexed';
                    END IF;
                ELSIF r.kind = 'evaluate' THEN
                    v_target_id := r.payload_json->>'run_id';
                    IF v_target_id IS NOT NULL AND v_target_id ~ '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$' THEN
                        UPDATE public.evaluation_runs
                        SET status = 'failed'
                        WHERE id = v_target_id
                          AND workspace_id = r.workspace_id
                          AND status != 'complete';
                    END IF;
                ELSIF r.kind = 'experiment' THEN
                    v_target_id := r.payload_json->>'experiment_id';
                    IF v_target_id IS NOT NULL AND v_target_id ~ '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$' THEN
                        UPDATE public.experiments
                        SET status = 'failed'
                        WHERE id = v_target_id
                          AND workspace_id = r.workspace_id
                          AND status != 'complete';
                    END IF;
                END IF;
            EXCEPTION WHEN OTHERS THEN
                UPDATE public.jobs
                SET last_error = 'Domain sync error: ' || SQLERRM,
                    detail = 'Execution timed out; domain sync failed: ' || SQLERRM
                WHERE id = r.id;
            END;
        END IF;

        reaped_id := r.id;
        reaped_workspace_id := r.workspace_id;
        reaped_kind := r.kind;
        final_status := 'failed';
        RETURN NEXT;
    END LOOP;
END;
$$;
""")
        )

        op.execute(
            sa.text("""
CREATE OR REPLACE FUNCTION public.manual_retry_job(
    p_job_id TEXT,
    p_workspace_id TEXT
)
RETURNS TABLE (
    id VARCHAR(36),
    status VARCHAR(24),
    cycle_attempts INT,
    manual_retry_count INT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    v_now TIMESTAMPTZ := clock_timestamp();
    v_caller_ws TEXT;
    r RECORD;
    v_target_id VARCHAR(36);
BEGIN
    v_caller_ws := current_setting('app.workspace_id', true);
    IF v_caller_ws IS NULL OR v_caller_ws = '' THEN
        RAISE EXCEPTION 'Workspace context not set (app.workspace_id is missing)';
    END IF;
    IF v_caller_ws != p_workspace_id THEN
        RAISE EXCEPTION 'Workspace authorization failed: context=% target=%', v_caller_ws, p_workspace_id;
    END IF;

    SELECT j.id, j.kind, j.payload_json, j.manual_retry_count
    INTO r
    FROM public.jobs j
    WHERE j.id = p_job_id
      AND j.workspace_id = p_workspace_id
      AND j.status = 'failed'
    FOR UPDATE;

    IF NOT FOUND THEN
        RETURN;
    END IF;

    UPDATE public.jobs
    SET status = 'queued',
        detail = 'Manually retried by operator',
        cycle_attempts = 0,
        manual_retry_count = r.manual_retry_count + 1,
        lease_token = NULL,
        lease_expires_at = NULL,
        worker_id = NULL,
        scheduled_at = v_now,
        last_error = NULL,
        updated_at = v_now
    WHERE public.jobs.id = r.id;

    IF r.payload_json IS NOT NULL THEN
        IF r.kind = 'index' THEN
            v_target_id := r.payload_json->>'document_id';
            IF v_target_id IS NOT NULL AND v_target_id ~ '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$' THEN
                UPDATE public.documents
                SET status = 'queued'
                WHERE public.documents.id = v_target_id AND workspace_id = p_workspace_id;
            END IF;
        ELSIF r.kind = 'evaluate' THEN
            v_target_id := r.payload_json->>'run_id';
            IF v_target_id IS NOT NULL AND v_target_id ~ '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$' THEN
                UPDATE public.evaluation_runs
                SET status = 'queued'
                WHERE public.evaluation_runs.id = v_target_id AND workspace_id = p_workspace_id;
            END IF;
        ELSIF r.kind = 'experiment' THEN
            v_target_id := r.payload_json->>'experiment_id';
            IF v_target_id IS NOT NULL AND v_target_id ~ '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$' THEN
                UPDATE public.experiments
                SET status = 'queued'
                WHERE public.experiments.id = v_target_id AND workspace_id = p_workspace_id;
            END IF;
        END IF;
    END IF;

    RETURN QUERY SELECT
        r.id,
        'queued'::VARCHAR(24),
        0::INT,
        (r.manual_retry_count + 1)::INT;
END;
$$;
""")
        )

        # 7. Function ownership assignment (two-owner least-privilege architecture)
        op.execute(
            sa.text("""
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_api_queue_owner') OR
       NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_worker_queue_owner') THEN
        RAISE EXCEPTION 'Migration prerequisite failure: Required roles rag_api_queue_owner and rag_worker_queue_owner do not exist. Execute deploy/sql/01_worker_roles_and_privileges.sql first.';
    END IF;

    -- Temporary role membership and schema CREATE required for ownership transfer in PostgreSQL 16+
    EXECUTE format('GRANT rag_api_queue_owner, rag_worker_queue_owner TO %I', current_user);
    GRANT CREATE ON SCHEMA public TO rag_api_queue_owner, rag_worker_queue_owner;

    -- API queue functions: owned by rag_api_queue_owner (strictly under tenant RLS, zero bypass)
    ALTER FUNCTION public.enqueue_job(TEXT, TEXT, JSONB, INT, TEXT) OWNER TO rag_api_queue_owner;
    ALTER FUNCTION public.manual_retry_job(TEXT, TEXT) OWNER TO rag_api_queue_owner;

    -- Worker queue functions: owned by rag_worker_queue_owner (scoped permissive policies)
    ALTER FUNCTION public.claim_next_job(TEXT, INT) OWNER TO rag_worker_queue_owner;
    ALTER FUNCTION public.reap_expired_jobs(INT, INT) OWNER TO rag_worker_queue_owner;

    -- Immediately revoke temporary CREATE privileges
    REVOKE CREATE ON SCHEMA public FROM rag_api_queue_owner, rag_worker_queue_owner;

    -- Worker queue owner permissive policies on queue and domain tables
    DROP POLICY IF EXISTS worker_queue_owner_jobs ON public.jobs;
    CREATE POLICY worker_queue_owner_jobs ON public.jobs
        FOR ALL TO rag_worker_queue_owner USING (true) WITH CHECK (true);

    DROP POLICY IF EXISTS worker_queue_owner_attempts ON public.job_attempts;
    CREATE POLICY worker_queue_owner_attempts ON public.job_attempts
        FOR ALL TO rag_worker_queue_owner USING (true) WITH CHECK (true);

    DROP POLICY IF EXISTS worker_queue_owner_documents ON public.documents;
    CREATE POLICY worker_queue_owner_documents ON public.documents
        FOR ALL TO rag_worker_queue_owner USING (true) WITH CHECK (true);

    DROP POLICY IF EXISTS worker_queue_owner_runs ON public.evaluation_runs;
    CREATE POLICY worker_queue_owner_runs ON public.evaluation_runs
        FOR ALL TO rag_worker_queue_owner USING (true) WITH CHECK (true);

    DROP POLICY IF EXISTS worker_queue_owner_experiments ON public.experiments;
    CREATE POLICY worker_queue_owner_experiments ON public.experiments
        FOR ALL TO rag_worker_queue_owner USING (true) WITH CHECK (true);
END
$$;
""")
        )

        # 8. Function ACL lockdown (fail-closed immediately after creation)
        op.execute(
            sa.text("""
-- Revoke default execution from PUBLIC unconditionally
REVOKE ALL ON FUNCTION public.enqueue_job(TEXT, TEXT, JSONB, INT, TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.claim_next_job(TEXT, INT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.reap_expired_jobs(INT, INT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.manual_retry_job(TEXT, TEXT) FROM PUBLIC;

-- Revoke from anon and authenticated if present in Supabase
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        REVOKE ALL ON FUNCTION public.enqueue_job(TEXT, TEXT, JSONB, INT, TEXT) FROM anon;
        REVOKE ALL ON FUNCTION public.claim_next_job(TEXT, INT) FROM anon;
        REVOKE ALL ON FUNCTION public.reap_expired_jobs(INT, INT) FROM anon;
        REVOKE ALL ON FUNCTION public.manual_retry_job(TEXT, TEXT) FROM anon;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
        REVOKE ALL ON FUNCTION public.enqueue_job(TEXT, TEXT, JSONB, INT, TEXT) FROM authenticated;
        REVOKE ALL ON FUNCTION public.claim_next_job(TEXT, INT) FROM authenticated;
        REVOKE ALL ON FUNCTION public.reap_expired_jobs(INT, INT) FROM authenticated;
        REVOKE ALL ON FUNCTION public.manual_retry_job(TEXT, TEXT) FROM authenticated;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_api') THEN
        REVOKE ALL ON FUNCTION public.claim_next_job(TEXT, INT) FROM rag_api;
        REVOKE ALL ON FUNCTION public.reap_expired_jobs(INT, INT) FROM rag_api;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_worker') THEN
        REVOKE ALL ON FUNCTION public.enqueue_job(TEXT, TEXT, JSONB, INT, TEXT) FROM rag_worker;
        REVOKE ALL ON FUNCTION public.manual_retry_job(TEXT, TEXT) FROM rag_worker;
    END IF;
END
$$;

-- Grant execution strictly to authorized application roles
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_worker') THEN
        GRANT EXECUTE ON FUNCTION public.claim_next_job(TEXT, INT) TO rag_worker;
        GRANT EXECUTE ON FUNCTION public.reap_expired_jobs(INT, INT) TO rag_worker;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_api') THEN
        GRANT EXECUTE ON FUNCTION public.manual_retry_job(TEXT, TEXT) TO rag_api;
        GRANT EXECUTE ON FUNCTION public.enqueue_job(TEXT, TEXT, JSONB, INT, TEXT) TO rag_api;
    END IF;
END
$$;
""")
        )


def downgrade() -> None:
    bind = op.get_bind()
    is_pg = bind.dialect.name == "postgresql"

    if is_pg:
        op.execute(sa.text('DROP POLICY IF EXISTS worker_queue_owner_experiments ON "experiments"'))
        op.execute(sa.text('DROP POLICY IF EXISTS worker_queue_owner_runs ON "evaluation_runs"'))
        op.execute(sa.text('DROP POLICY IF EXISTS worker_queue_owner_documents ON "documents"'))
        op.execute(sa.text('DROP POLICY IF EXISTS worker_queue_owner_attempts ON "job_attempts"'))
        op.execute(sa.text('DROP POLICY IF EXISTS worker_queue_owner_jobs ON "jobs"'))
        op.execute(
            sa.text("DROP FUNCTION IF EXISTS public.enqueue_job(TEXT, TEXT, JSONB, INT, TEXT)")
        )
        op.execute(sa.text("DROP FUNCTION IF EXISTS public.manual_retry_job(TEXT, TEXT)"))
        op.execute(sa.text("DROP FUNCTION IF EXISTS public.reap_expired_jobs(INT, INT)"))
        op.execute(sa.text("DROP FUNCTION IF EXISTS public.claim_next_job(TEXT, INT)"))

    op.drop_index("ix_jobs_lease_token", table_name="jobs")
    op.drop_index("ix_jobs_queue_claim", table_name="jobs")
    op.drop_table("job_attempts")

    for col in (
        "completed_at",
        "updated_at",
        "started_at",
        "last_error",
        "scheduled_at",
        "manual_retry_count",
        "max_attempts",
        "cumulative_attempts",
        "cycle_attempts",
        "lease_expires_at",
        "lease_generation",
        "lease_token",
        "worker_id",
        "payload_json",
    ):
        op.drop_column("jobs", col)
