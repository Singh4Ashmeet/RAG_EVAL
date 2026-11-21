from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

os.environ["SKIP_SCHEMA_INIT"] = "true"
os.environ["SKIP_RLS_SETUP"] = "true"

from apps.api.main import Base  # noqa: E402

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

is_prod = os.getenv("RENDER") == "true" or os.getenv("ENVIRONMENT") == "production"

migration_url = os.getenv("MIGRATION_DATABASE_URL")
database_url = os.getenv("DATABASE_URL")

if is_prod:
    if not migration_url:
        raise RuntimeError(
            "Production migration error: MIGRATION_DATABASE_URL is required to run migrations "
            "in production. Alembic DDL migrations must be executed out-of-band by an authorized "
            "operator with administrative credentials. Do not execute migrations using the "
            "restricted application runtime credentials (DATABASE_URL)."
        )
    if "rag_api" in migration_url or (
        database_url and migration_url.strip() == database_url.strip()
    ):
        raise RuntimeError(
            "Production migration error: MIGRATION_DATABASE_URL cannot use restricted "
            "application role 'rag_api'. Dedicated administrative/migration credentials "
            "are required for DDL operations."
        )
    effective_url = migration_url
else:
    # In local development/testing, use MIGRATION_DATABASE_URL if provided, else DATABASE_URL
    effective_url = migration_url or database_url

if effective_url:
    if effective_url.startswith("postgres://"):
        effective_url = "postgresql+psycopg://" + effective_url.removeprefix("postgres://")
    elif effective_url.startswith("postgresql://"):
        effective_url = "postgresql+psycopg://" + effective_url.removeprefix("postgresql://")
    config.set_main_option("sqlalchemy.url", effective_url.replace("%", "%%"))

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
