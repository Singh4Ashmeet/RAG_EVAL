-- RAG_EVAL: Database Roles, Domain Privileges, and Bootstrap Prerequisites
-- Deployment artifact for production PostgreSQL (Supabase).
-- To be executed out-of-band by the database administrator (postgres).

-- 1. Create Application and Function Owner Roles
-- Role attributes are explicitly verified/normalized for least privilege
DO $$
DECLARE
    v_current_user TEXT := current_user;
BEGIN
    -- rag_api_queue_owner: NOLOGIN, NOINHERIT
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rag_api_queue_owner') THEN
        CREATE ROLE rag_api_queue_owner WITH NOLOGIN NOINHERIT;
    ELSE
        ALTER ROLE rag_api_queue_owner WITH NOLOGIN NOINHERIT;
    END IF;

    -- rag_worker_queue_owner: NOLOGIN, NOINHERIT
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rag_worker_queue_owner') THEN
        CREATE ROLE rag_worker_queue_owner WITH NOLOGIN NOINHERIT;
    ELSE
        ALTER ROLE rag_worker_queue_owner WITH NOLOGIN NOINHERIT;
    END IF;

    -- rag_api: LOGIN, NOINHERIT
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rag_api') THEN
        CREATE ROLE rag_api WITH LOGIN NOINHERIT;
    ELSE
        ALTER ROLE rag_api WITH LOGIN NOINHERIT;
    END IF;

    -- rag_worker: LOGIN, NOINHERIT
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rag_worker') THEN
        CREATE ROLE rag_worker WITH LOGIN NOINHERIT;
    ELSE
        ALTER ROLE rag_worker WITH LOGIN NOINHERIT;
    END IF;

    -- In PostgreSQL 16+, transferring ownership requires the executing user
    -- to be able to SET ROLE to the new owner. Grant membership to current user.
    EXECUTE format('GRANT rag_api_queue_owner, rag_worker_queue_owner TO %I', v_current_user);
END
$$;

-- 2. Schema and Connection Privileges
DO $$
BEGIN
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO rag_api, rag_worker', current_database());
END
$$;
GRANT USAGE ON SCHEMA public TO rag_api, rag_worker, rag_api_queue_owner, rag_worker_queue_owner;

-- Account bootstrap RLS: an unauthenticated request has no workspace ID yet.
-- The API sets these transaction-local values only after a provider verifies
-- the subject and email; new accounts must also have an invitation.
ALTER TABLE public.accounts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.accounts FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS account_identity_isolation ON public.accounts;
DROP POLICY IF EXISTS account_invited_creation ON public.accounts;
DROP POLICY IF EXISTS account_self_update ON public.accounts;
DROP POLICY IF EXISTS account_self_delete ON public.accounts;
CREATE POLICY account_identity_isolation ON public.accounts
    FOR SELECT TO rag_api
    USING (
        id = current_setting('app.workspace_id', true)
        OR github_id = current_setting('app.auth_subject', true)
        OR email = current_setting('app.auth_email', true)
    );
CREATE POLICY account_invited_creation ON public.accounts
    FOR INSERT TO rag_api
    WITH CHECK (
        github_id = current_setting('app.auth_subject', true)
        AND email = current_setting('app.auth_email', true)
        AND EXISTS (
            SELECT 1 FROM public.invitations
            WHERE invitations.email = accounts.email
        )
    );
CREATE POLICY account_self_update ON public.accounts
    FOR UPDATE TO rag_api
    USING (
        id = current_setting('app.workspace_id', true)
        OR github_id = current_setting('app.auth_subject', true)
        OR email = current_setting('app.auth_email', true)
    )
    WITH CHECK (
        id = current_setting('app.workspace_id', true)
        OR github_id = current_setting('app.auth_subject', true)
        OR email = current_setting('app.auth_email', true)
    );
CREATE POLICY account_self_delete ON public.accounts
    FOR DELETE TO rag_api
    USING (id = current_setting('app.workspace_id', true));

-- 3. Base Table Grants for rag_api
-- General domain tables: full CRUD
GRANT SELECT, INSERT, UPDATE, DELETE ON public.accounts TO rag_api;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.sessions TO rag_api;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.documents TO rag_api;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.corpus_elements TO rag_api;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.datasets TO rag_api;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.dataset_revisions TO rag_api;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.evaluation_runs TO rag_api;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.experiments TO rag_api;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.provider_credentials TO rag_api;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.invitations TO rag_api;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO rag_api;

-- Jobs table grants for rag_api:
-- API can SELECT jobs (filtered by workspace RLS)
GRANT SELECT ON public.jobs TO rag_api;
-- API CANNOT INSERT, UPDATE, or DELETE jobs directly.
-- All job creation goes through the SECURITY DEFINER enqueue_job() function.
REVOKE INSERT, UPDATE, DELETE ON public.jobs FROM rag_api;

-- 4. Base Table Grants for rag_worker under Row-Level Security
GRANT SELECT, UPDATE ON public.jobs TO rag_worker;
REVOKE DELETE ON public.jobs FROM rag_worker;
GRANT SELECT, UPDATE ON public.documents TO rag_worker;
-- Corpus elements: worker reads and updates, but does NOT insert (no sequence-backed PK)
REVOKE INSERT, DELETE ON public.corpus_elements FROM rag_worker;
GRANT SELECT, UPDATE ON public.corpus_elements TO rag_worker;
GRANT SELECT, UPDATE ON public.evaluation_runs TO rag_worker;
GRANT SELECT, UPDATE ON public.experiments TO rag_worker;
GRANT SELECT ON public.datasets TO rag_worker;
GRANT SELECT ON public.dataset_revisions TO rag_worker;
GRANT SELECT ON public.accounts TO rag_worker;

-- 4b. Table Grants for rag_api_queue_owner (API Queue Execution Identity)
-- Dedicated non-login role owns enqueue_job and manual_retry_job.
-- Operates strictly under standard tenant RLS (workspace_isolation). Zero bypass policies.
GRANT SELECT, INSERT, UPDATE ON public.jobs TO rag_api_queue_owner;
GRANT UPDATE ON public.documents TO rag_api_queue_owner;
GRANT UPDATE ON public.evaluation_runs TO rag_api_queue_owner;
GRANT UPDATE ON public.experiments TO rag_api_queue_owner;

-- 4c. Table Grants and Permissive RLS Policies for rag_worker_queue_owner (Worker Execution Identity)
-- Dedicated non-login role owns claim_next_job and reap_expired_jobs.
-- Holds table-scoped permissive policies to inspect and reap across tenant workspaces.
GRANT SELECT, UPDATE ON public.jobs TO rag_worker_queue_owner;
GRANT SELECT, UPDATE ON public.documents TO rag_worker_queue_owner;
GRANT SELECT, UPDATE ON public.evaluation_runs TO rag_worker_queue_owner;
GRANT SELECT, UPDATE ON public.experiments TO rag_worker_queue_owner;

DROP POLICY IF EXISTS worker_queue_owner_jobs ON public.jobs;
CREATE POLICY worker_queue_owner_jobs ON public.jobs
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

-- 4d. Conditional grants for job_attempts (applied if table already exists;
-- also unconditionally granted upon table creation in Alembic migration 20261109_0003)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'job_attempts') THEN
        GRANT SELECT ON public.job_attempts TO rag_api;
        REVOKE INSERT, UPDATE, DELETE ON public.job_attempts FROM rag_api;

        GRANT SELECT, INSERT, UPDATE ON public.job_attempts TO rag_worker;
        REVOKE DELETE ON public.job_attempts FROM rag_worker;

        -- Least privilege: rag_api_queue_owner strictly does not access job_attempts
        REVOKE ALL ON public.job_attempts FROM rag_api_queue_owner;
        GRANT SELECT, INSERT, UPDATE ON public.job_attempts TO rag_worker_queue_owner;

        DROP POLICY IF EXISTS worker_queue_owner_attempts ON public.job_attempts;
        CREATE POLICY worker_queue_owner_attempts ON public.job_attempts
            FOR ALL TO rag_worker_queue_owner USING (true) WITH CHECK (true);
    END IF;
END
$$;

-- Post-Provisioning Verification Assertions
DO $$
DECLARE
    v_missing_roles TEXT[] := '{}';
    v_owner_enq TEXT;
    v_owner_ret TEXT;
    v_owner_clm TEXT;
    v_owner_rep TEXT;
    v_proc_oid OID;
    v_proacl ACLITEM[];
    v_has_public_exec BOOLEAN;
BEGIN
    -- 1. Assert all 4 required roles exist
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rag_api_queue_owner') THEN
        v_missing_roles := array_append(v_missing_roles, 'rag_api_queue_owner');
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rag_worker_queue_owner') THEN
        v_missing_roles := array_append(v_missing_roles, 'rag_worker_queue_owner');
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rag_api') THEN
        v_missing_roles := array_append(v_missing_roles, 'rag_api');
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rag_worker') THEN
        v_missing_roles := array_append(v_missing_roles, 'rag_worker');
    END IF;

    IF array_length(v_missing_roles, 1) > 0 THEN
        RAISE EXCEPTION 'Role bootstrap assertion failed: Missing roles: %', array_to_string(v_missing_roles, ', ');
    END IF;

    -- 2. Verify role attributes (least privilege)
    -- rag_api_queue_owner: NOLOGIN, NOINHERIT, NOSUPERUSER, NOCREATEDB, NOCREATEROLE, NOREPLICATION, NOBYPASSRLS
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_api_queue_owner' AND (rolsuper OR rolbypassrls OR rolcanlogin OR rolcreaterole OR rolcreatedb OR rolreplication)) THEN
        RAISE EXCEPTION 'Role rag_api_queue_owner has unexpected attributes: must be NOLOGIN, NOINHERIT, NOSUPERUSER, NOCREATEDB, NOCREATEROLE, NOREPLICATION, NOBYPASSRLS';
    END IF;

    -- rag_worker_queue_owner: NOLOGIN, NOINHERIT, NOSUPERUSER, NOCREATEDB, NOCREATEROLE, NOREPLICATION, NOBYPASSRLS
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_worker_queue_owner' AND (rolsuper OR rolbypassrls OR rolcanlogin OR rolcreaterole OR rolcreatedb OR rolreplication)) THEN
        RAISE EXCEPTION 'Role rag_worker_queue_owner has unexpected attributes: must be NOLOGIN, NOINHERIT, NOSUPERUSER, NOCREATEDB, NOCREATEROLE, NOREPLICATION, NOBYPASSRLS';
    END IF;

    -- rag_api: LOGIN, NOINHERIT, NOSUPERUSER, NOCREATEROLE, NOCREATEDB, NOREPLICATION, NOBYPASSRLS
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_api' AND (NOT rolcanlogin OR rolsuper OR rolbypassrls OR rolcreaterole OR rolcreatedb OR rolreplication)) THEN
        RAISE EXCEPTION 'Role rag_api has unexpected attributes: must be LOGIN, NOINHERIT, NOSUPERUSER, NOCREATEROLE, NOCREATEDB, NOREPLICATION, NOBYPASSRLS';
    END IF;

    -- rag_worker: LOGIN, NOINHERIT, NOSUPERUSER, NOCREATEROLE, NOCREATEDB, NOREPLICATION, NOBYPASSRLS
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_worker' AND (NOT rolcanlogin OR rolsuper OR rolbypassrls OR rolcreaterole OR rolcreatedb OR rolreplication)) THEN
        RAISE EXCEPTION 'Role rag_worker has unexpected attributes: must be LOGIN, NOINHERIT, NOSUPERUSER, NOCREATEROLE, NOCREATEDB, NOREPLICATION, NOBYPASSRLS';
    END IF;

    -- 3. Verify schema CREATE is NOT granted to owner roles
    IF has_schema_privilege('rag_api_queue_owner', 'public', 'CREATE') THEN
        RAISE EXCEPTION 'Security assertion failed: rag_api_queue_owner unexpectedly retains CREATE on schema public';
    END IF;
    IF has_schema_privilege('rag_worker_queue_owner', 'public', 'CREATE') THEN
        RAISE EXCEPTION 'Security assertion failed: rag_worker_queue_owner unexpectedly retains CREATE on schema public';
    END IF;

    -- 4. Verify role memberships for unintended privilege escalation
    -- Queue owners should not be members of privileged roles
    IF EXISTS (SELECT 1 FROM pg_auth_members m JOIN pg_roles r ON m.roleid = r.oid WHERE m.member = 'rag_api_queue_owner'::regrole AND r.rolsuper) THEN
        RAISE EXCEPTION 'Security assertion failed: rag_api_queue_owner is member of superuser role';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_auth_members m JOIN pg_roles r ON m.roleid = r.oid WHERE m.member = 'rag_worker_queue_owner'::regrole AND r.rolsuper) THEN
        RAISE EXCEPTION 'Security assertion failed: rag_worker_queue_owner is member of superuser role';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_auth_members m JOIN pg_roles r ON m.roleid = r.oid WHERE m.member = 'rag_api_queue_owner'::regrole AND r.rolcreaterole) THEN
        RAISE EXCEPTION 'Security assertion failed: rag_api_queue_owner is member of createrole role';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_auth_members m JOIN pg_roles r ON m.roleid = r.oid WHERE m.member = 'rag_worker_queue_owner'::regrole AND r.rolcreaterole) THEN
        RAISE EXCEPTION 'Security assertion failed: rag_worker_queue_owner is member of createrole role';
    END IF;

    -- 5. Verify rag_api_queue_owner holds strictly NO privileges on job_attempts
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'job_attempts') THEN
        IF has_table_privilege('rag_api_queue_owner', 'public.job_attempts', 'SELECT') OR
           has_table_privilege('rag_api_queue_owner', 'public.job_attempts', 'INSERT') OR
           has_table_privilege('rag_api_queue_owner', 'public.job_attempts', 'UPDATE') OR
           has_table_privilege('rag_api_queue_owner', 'public.job_attempts', 'DELETE') THEN
            RAISE EXCEPTION 'Security assertion failed: rag_api_queue_owner unexpectedly holds privileges on job_attempts';
        END IF;
    END IF;

    -- 6. Verify function ownership and exact signatures for ALL FOUR functions independently
    -- enqueue_job(TEXT, TEXT, JSONB, INT, TEXT)
    v_proc_oid := to_regprocedure('public.enqueue_job(text, text, jsonb, integer, text)');
    IF v_proc_oid IS NULL THEN
        RAISE EXCEPTION 'Function ownership verification failed: public.enqueue_job(text, text, jsonb, integer, text) does not exist';
    END IF;
    SELECT p.proowner::regrole::text INTO v_owner_enq FROM pg_proc p WHERE p.oid = v_proc_oid;
    IF v_owner_enq != 'rag_api_queue_owner' THEN
        RAISE EXCEPTION 'Ownership verification failed: enqueue_job must be owned by rag_api_queue_owner (got %)', v_owner_enq;
    END IF;

    -- manual_retry_job(TEXT, TEXT)
    v_proc_oid := to_regprocedure('public.manual_retry_job(text, text)');
    IF v_proc_oid IS NULL THEN
        RAISE EXCEPTION 'Function ownership verification failed: public.manual_retry_job(text, text) does not exist';
    END IF;
    SELECT p.proowner::regrole::text INTO v_owner_ret FROM pg_proc p WHERE p.oid = v_proc_oid;
    IF v_owner_ret != 'rag_api_queue_owner' THEN
        RAISE EXCEPTION 'Ownership verification failed: manual_retry_job must be owned by rag_api_queue_owner (got %)', v_owner_ret;
    END IF;

    -- claim_next_job(TEXT, INT)
    v_proc_oid := to_regprocedure('public.claim_next_job(text, integer)');
    IF v_proc_oid IS NULL THEN
        RAISE EXCEPTION 'Function ownership verification failed: public.claim_next_job(text, integer) does not exist';
    END IF;
    SELECT p.proowner::regrole::text INTO v_owner_clm FROM pg_proc p WHERE p.oid = v_proc_oid;
    IF v_owner_clm != 'rag_worker_queue_owner' THEN
        RAISE EXCEPTION 'Ownership verification failed: claim_next_job must be owned by rag_worker_queue_owner (got %)', v_owner_clm;
    END IF;

    -- reap_expired_jobs(INT, INT)
    v_proc_oid := to_regprocedure('public.reap_expired_jobs(integer, integer)');
    IF v_proc_oid IS NULL THEN
        RAISE EXCEPTION 'Function ownership verification failed: public.reap_expired_jobs(integer, integer) does not exist';
    END IF;
    SELECT p.proowner::regrole::text INTO v_owner_rep FROM pg_proc p WHERE p.oid = v_proc_oid;
    IF v_owner_rep != 'rag_worker_queue_owner' THEN
        RAISE EXCEPTION 'Ownership verification failed: reap_expired_jobs must be owned by rag_worker_queue_owner (got %)', v_owner_rep;
    END IF;

    -- 6. Verify PUBLIC has NO EXECUTE on ALL FOUR queue functions using ACL inspection
    -- PUBLIC is represented by grantee OID 0 in pg_proc.proacl
    -- Use aclexplode() which returns (grantor oid, grantee oid, is_grantable bool, privilege_type text)
    -- Check enqueue_job
    IF EXISTS (
        SELECT 1 FROM aclexplode(
            COALESCE(
                (SELECT proacl FROM pg_proc WHERE oid = to_regprocedure('public.enqueue_job(text, text, jsonb, integer, text)')),
                acldefault('f', (SELECT proowner FROM pg_proc WHERE oid = to_regprocedure('public.enqueue_job(text, text, jsonb, integer, text)')))
            )
        ) WHERE grantee = 0 AND privilege_type = 'EXECUTE'
    ) THEN
        RAISE EXCEPTION 'Security assertion failed: PUBLIC has EXECUTE on enqueue_job';
    END IF;

    -- Check claim_next_job
    IF EXISTS (
        SELECT 1 FROM aclexplode(
            COALESCE(
                (SELECT proacl FROM pg_proc WHERE oid = to_regprocedure('public.claim_next_job(text, integer)')),
                acldefault('f', (SELECT proowner FROM pg_proc WHERE oid = to_regprocedure('public.claim_next_job(text, integer)')))
            )
        ) WHERE grantee = 0 AND privilege_type = 'EXECUTE'
    ) THEN
        RAISE EXCEPTION 'Security assertion failed: PUBLIC has EXECUTE on claim_next_job';
    END IF;

    -- Check reap_expired_jobs
    IF EXISTS (
        SELECT 1 FROM aclexplode(
            COALESCE(
                (SELECT proacl FROM pg_proc WHERE oid = to_regprocedure('public.reap_expired_jobs(integer, integer)')),
                acldefault('f', (SELECT proowner FROM pg_proc WHERE oid = to_regprocedure('public.reap_expired_jobs(integer, integer)')))
            )
        ) WHERE grantee = 0 AND privilege_type = 'EXECUTE'
    ) THEN
        RAISE EXCEPTION 'Security assertion failed: PUBLIC has EXECUTE on reap_expired_jobs';
    END IF;

    -- Check manual_retry_job
    IF EXISTS (
        SELECT 1 FROM aclexplode(
            COALESCE(
                (SELECT proacl FROM pg_proc WHERE oid = to_regprocedure('public.manual_retry_job(text, text)')),
                acldefault('f', (SELECT proowner FROM pg_proc WHERE oid = to_regprocedure('public.manual_retry_job(text, text)')))
            )
        ) WHERE grantee = 0 AND privilege_type = 'EXECUTE'
    ) THEN
        RAISE EXCEPTION 'Security assertion failed: PUBLIC has EXECUTE on manual_retry_job';
    END IF;

    -- 8. Verify worker permissions on corpus_elements are SELECT and UPDATE only
    IF has_table_privilege('rag_worker', 'public.corpus_elements', 'INSERT') THEN
        RAISE EXCEPTION 'Security assertion failed: rag_worker must not have INSERT on corpus_elements';
    END IF;
    IF has_table_privilege('rag_worker', 'public.corpus_elements', 'DELETE') THEN
        RAISE EXCEPTION 'Security assertion failed: rag_worker must not have DELETE on corpus_elements';
    END IF;
    IF NOT has_table_privilege('rag_worker', 'public.corpus_elements', 'SELECT') THEN
        RAISE EXCEPTION 'Security assertion failed: rag_worker must have SELECT on corpus_elements';
    END IF;
    IF NOT has_table_privilege('rag_worker', 'public.corpus_elements', 'UPDATE') THEN
        RAISE EXCEPTION 'Security assertion failed: rag_worker must have UPDATE on corpus_elements';
    END IF;

    RAISE NOTICE 'RAG_EVAL: Post-bootstrap verification complete. All roles, functions, and privileges verified.';
END
$$;
