"""Centralized application configuration with environment-aware validation."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # =========================================================================
    # Application Environment
    # =========================================================================
    app_env: Literal["development", "production", "test"] = Field(
        default="development",
        validation_alias="APP_ENV",
        description="Application environment mode",
    )
    app_debug: bool = Field(
        default=False,
        validation_alias="APP_DEBUG",
        description="Enable debug mode (only for development)",
    )

    @field_validator("app_debug", mode="after")
    @classmethod
    def validate_debug_in_production(cls, v: bool, info) -> bool:
        if info.data.get("app_env") == "production" and v:
            raise ValueError("APP_DEBUG must be false in production")
        return v

    # =========================================================================
    # Database
    # =========================================================================
    database_url: str = Field(
        default="sqlite:///./rag_eval_hosted.db",
        validation_alias="DATABASE_URL",
        description="SQLAlchemy database connection URL",
    )

    # =========================================================================
    # Redis / Queue
    # =========================================================================
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        validation_alias="REDIS_URL",
        description="Redis connection URL for RQ job queue",
    )
    rq_job_timeout: int = Field(
        default=600,
        validation_alias="RQ_JOB_TIMEOUT",
        ge=30,
        le=3600,
        description="RQ job timeout in seconds",
    )
    rq_result_ttl: int = Field(
        default=86400,
        validation_alias="RQ_RESULT_TTL",
        ge=60,
        description="RQ successful job result TTL in seconds",
    )
    rq_failure_ttl: int = Field(
        default=604800,
        validation_alias="RQ_FAILURE_TTL",
        ge=60,
        description="RQ failed job retention TTL in seconds",
    )

    @field_validator("redis_url", mode="after")
    @classmethod
    def validate_redis_url_in_production(cls, v: str, info) -> str:
        if info.data.get("app_env") == "production" and v.startswith("redis://localhost"):
            raise ValueError(
                "REDIS_URL must not point to localhost in production. "
                "Configure a reachable Redis service."
            )
        return v

    # =========================================================================
    # Inference Providers
    # =========================================================================
    embedding_provider: Literal["ollama"] = Field(
        default="ollama",
        validation_alias="EMBEDDING_PROVIDER",
        description="Embedding provider (currently only ollama)",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        validation_alias="EMBEDDING_MODEL",
        description="Embedding model name",
    )
    embedding_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="EMBEDDING_BASE_URL",
        description="Embedding service base URL",
    )
    embedding_api_key: str = Field(
        default="",
        validation_alias="EMBEDDING_API_KEY",
        description="Embedding service API key (if required)",
    )

    chat_provider: Literal["ollama", "gemini", "groq"] = Field(
        default="ollama",
        validation_alias="CHAT_PROVIDER",
        description="Chat completion provider",
    )
    chat_model: str = Field(
        default="llama3.2:3b",
        validation_alias="CHAT_MODEL",
        description="Chat model name",
    )
    chat_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="CHAT_BASE_URL",
        description="Chat service base URL (for Ollama)",
    )
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com",
        validation_alias="GEMINI_BASE_URL",
        description="Gemini API base URL",
    )
    groq_base_url: str = Field(
        default="https://api.groq.com",
        validation_alias="GROQ_BASE_URL",
        description="Groq API base URL",
    )
    gemini_api_key: str = Field(
        default="",
        validation_alias="GEMINI_API_KEY",
        description="Gemini API key (if required)",
    )
    groq_api_key: str = Field(
        default="",
        validation_alias="GROQ_API_KEY",
        description="Groq API key (if required)",
    )
    chat_api_key: str = Field(
        default="",
        validation_alias="CHAT_API_KEY",
        description="Chat service API key (if required)",
    )

    gemini_chat_model: str = Field(
        default="gemini-2.5-flash",
        validation_alias="GEMINI_CHAT_MODEL",
    )
    groq_chat_model: str = Field(
        default="qwen/qwen3.8-27b",
        validation_alias="GROQ_CHAT_MODEL",
    )

    # =========================================================================
    # Provider Fallback
    # =========================================================================
    fallback_enabled: bool = Field(
        default=False,
        validation_alias="FALLBACK_ENABLED",
        description="Enable automatic fallback to alternative providers on failure",
    )
    fallback_providers: str = Field(
        default="",
        validation_alias="FALLBACK_PROVIDERS",
        description=(
            "Comma-separated list of fallback providers in priority order "
            "(e.g., 'gemini,groq,ollama')"
        ),
    )

    # =========================================================================
    # Provider Retry
    # =========================================================================
    retry_enabled: bool = Field(
        default=True,
        validation_alias="RETRY_ENABLED",
        description="Enable retry for transient provider failures",
    )
    retry_max_attempts: int = Field(
        default=3,
        validation_alias="RETRY_MAX_ATTEMPTS",
        ge=1,
        le=10,
        description="Maximum generation attempts per provider (including initial attempt)",
    )
    retry_base_delay_seconds: float = Field(
        default=1.0,
        validation_alias="RETRY_BASE_DELAY_SECONDS",
        ge=0.0,
        le=60.0,
        description="Initial backoff delay in seconds",
    )
    retry_max_delay_seconds: float = Field(
        default=10.0,
        validation_alias="RETRY_MAX_DELAY_SECONDS",
        ge=0.0,
        le=300.0,
        description="Maximum backoff delay cap in seconds",
    )
    retry_exponential_base: float = Field(
        default=2.0,
        validation_alias="RETRY_EXPONENTIAL_BASE",
        ge=1.0,
        le=5.0,
        description="Exponential base for backoff calculation",
    )
    retry_jitter_enabled: bool = Field(
        default=True,
        validation_alias="RETRY_JITTER_ENABLED",
        description="Apply jitter to prevent synchronized retry storms",
    )
    retry_timeout_seconds: float = Field(
        default=30.0,
        validation_alias="RETRY_TIMEOUT_SECONDS",
        ge=1.0,
        le=300.0,
        description="Maximum total time budget for all retry attempts per provider",
    )

    # =========================================================================
    # Provider Circuit Breaker
    # =========================================================================
    circuit_breaker_enabled: bool = Field(
        default=True,
        validation_alias="CIRCUIT_BREAKER_ENABLED",
        description="Enable circuit breaker for provider failure isolation",
    )
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        validation_alias="CIRCUIT_BREAKER_FAILURE_THRESHOLD",
        ge=1,
        le=100,
        description="Number of consecutive provider failures before opening circuit",
    )
    circuit_breaker_recovery_timeout_seconds: float = Field(
        default=30.0,
        validation_alias="CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS",
        ge=1.0,
        le=3600.0,
        description="Seconds to wait before attempting recovery (HALF_OPEN state)",
    )
    circuit_breaker_half_open_max_calls: int = Field(
        default=1,
        validation_alias="CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS",
        ge=1,
        le=10,
        description="Maximum probe calls allowed in HALF_OPEN state",
    )
    circuit_breaker_window_seconds: float = Field(
        default=60.0,
        validation_alias="CIRCUIT_BREAKER_WINDOW_SECONDS",
        ge=1.0,
        le=3600.0,
        description="Rolling window in seconds for failure counting",
    )

    @classmethod
    def validate_remote_urls_in_production(cls, v: str, info) -> str:
        if info.data.get("app_env") == "production" and v.startswith("http://localhost"):
            raise ValueError(
                "Inference base URLs must not point to localhost in production. "
                "Configure a reachable remote endpoint."
            )
        return v

    # =========================================================================
    # S3-Compatible Storage
    # =========================================================================
    storage_backend: Literal["local", "s3"] = Field(
        default="local",
        validation_alias="STORAGE_BACKEND",
        description="Storage backend for original documents",
    )
    s3_bucket: str = Field(
        default="",
        validation_alias="S3_BUCKET",
        description="S3 bucket name",
    )
    s3_region: str = Field(
        default="us-east-1",
        validation_alias="S3_REGION",
    )
    s3_endpoint_url: str = Field(
        default="",
        validation_alias="S3_ENDPOINT_URL",
        description="S3-compatible endpoint URL (for non-AWS providers)",
    )
    s3_access_key_id: str = Field(
        default="",
        validation_alias="S3_ACCESS_KEY_ID",
    )
    s3_secret_access_key: str = Field(
        default="",
        validation_alias="S3_SECRET_ACCESS_KEY",
    )
    s3_prefix: str = Field(
        default="rag-eval/",
        validation_alias="S3_PREFIX",
        description="Object key prefix within bucket",
    )

    @field_validator("s3_bucket", "s3_access_key_id", "s3_secret_access_key", mode="after")
    @classmethod
    def validate_s3_config_in_production(cls, v: str, info) -> str:
        if info.data.get("app_env") == "production" and info.data.get("storage_backend") == "s3":
            if not v:
                field_name = info.field_name
                raise ValueError(f"{field_name} is required when STORAGE_BACKEND=s3 in production")
        return v

    # =========================================================================
    # CORS & Security
    # =========================================================================
    cors_allowed_origins: str = Field(
        default="",
        validation_alias="CORS_ALLOWED_ORIGINS",
        description="Comma-separated list of allowed CORS origins",
    )

    def cors_origins_list(self) -> list[str]:
        if not self.cors_allowed_origins:
            return []
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]

    # =========================================================================
    # Credentials & Auth
    # =========================================================================
    credential_encryption_key: str = Field(
        default="",
        validation_alias="CREDENTIAL_ENCRYPTION_KEY",
        description="Fernet key for encrypting provider credentials",
    )
    invitation_admin_token: str = Field(
        default="",
        validation_alias="INVITATION_ADMIN_TOKEN",
    )
    cookie_secure: bool = Field(
        default=False,
        validation_alias="COOKIE_SECURE",
    )
    public_base_url: str = Field(
        default="http://localhost:5173",
        validation_alias="PUBLIC_BASE_URL",
    )
    auth_dev_mode: bool = Field(
        default=True,
        validation_alias="AUTH_DEV_MODE",
    )
    resend_api_key: str = Field(
        default="console",
        validation_alias="RESEND_API_KEY",
    )
    auth_email_from: str = Field(
        default="RAG_EVAL <sign-in@localhost>",
        validation_alias="AUTH_EMAIL_FROM",
    )
    github_client_id: str = Field(
        default="",
        validation_alias="GITHUB_CLIENT_ID",
    )
    github_client_secret: str = Field(
        default="",
        validation_alias="GITHUB_CLIENT_SECRET",
    )
    google_client_id: str = Field(
        default="",
        validation_alias="GOOGLE_CLIENT_ID",
    )
    google_client_secret: str = Field(
        default="",
        validation_alias="GOOGLE_CLIENT_SECRET",
    )
    skip_schema_init: str = Field(
        default="false",
        validation_alias="SKIP_SCHEMA_INIT",
    )
    skip_rls_setup: str = Field(
        default="false",
        validation_alias="SKIP_RLS_SETUP",
    )

    @field_validator("credential_encryption_key", mode="after")
    @classmethod
    def validate_credential_key(cls, v: str, info) -> str:
        if info.data.get("app_env") == "production" and not v:
            raise ValueError("CREDENTIAL_ENCRYPTION_KEY is required in production")
        return v

    @field_validator("invitation_admin_token", mode="after")
    @classmethod
    def validate_invitation_token(cls, v: str, info) -> str:
        if info.data.get("app_env") == "production" and not v:
            raise ValueError("INVITATION_ADMIN_TOKEN is required in production")
        return v

    @field_validator("cookie_secure", mode="after")
    @classmethod
    def validate_cookie_secure(cls, v: bool, info) -> bool:
        if info.data.get("app_env") == "production" and not v:
            raise ValueError("COOKIE_SECURE must be true in production (requires HTTPS)")
        return v

    @field_validator("public_base_url", mode="after")
    @classmethod
    def validate_public_base_url(cls, v: str, info) -> str:
        if info.data.get("app_env") == "production" and v.startswith("http://localhost"):
            raise ValueError("PUBLIC_BASE_URL must not be localhost in production")
        return v

    # =========================================================================
    # File Upload
    # =========================================================================
    max_upload_bytes: int = Field(
        default=26214400,
        validation_alias="MAX_UPLOAD_BYTES",
        ge=1024,
        le=104857600,
    )
    private_data_dir: str = Field(
        default="./private_data",
        validation_alias="PRIVATE_DATA_DIR",
    )

    # =========================================================================
    # Ollama (Legacy - kept for backward compatibility)
    # =========================================================================
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    ollama_embed_model: str = Field(
        default="nomic-embed-text",
        validation_alias="OLLAMA_EMBED_MODEL",
    )
    ollama_chat_model: str = Field(
        default="llama3.2:3b",
        validation_alias="OLLAMA_CHAT_MODEL",
    )
    gemini_chat_model_legacy: str = Field(
        default="gemini-3.8-flash",
        validation_alias="GEMINI_CHAT_MODEL",
    )
    groq_chat_model_legacy: str = Field(
        default="openai/gpt-oss-120b",
        validation_alias="GROQ_CHAT_MODEL",
    )

    # =========================================================================
    # Derived Properties
    # =========================================================================
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_test(self) -> bool:
        return self.app_env == "test"

    def validate_production_requirements(self) -> None:
        """Validate all production requirements. Raises ValidationError if not met."""
        if not self.is_production:
            return

        errors = []

        if self.database_url.startswith("sqlite"):
            errors.append("DATABASE_URL must use PostgreSQL in production")

        if self.redis_url.startswith("redis://localhost"):
            errors.append("REDIS_URL must not be localhost in production")

        if self.embedding_base_url.startswith("http://localhost"):
            errors.append("EMBEDDING_BASE_URL must not be localhost in production")

        if self.chat_base_url.startswith("http://localhost"):
            errors.append("CHAT_BASE_URL must not be localhost in production")

        if not self.credential_encryption_key:
            errors.append("CREDENTIAL_ENCRYPTION_KEY is required")

        if not self.invitation_admin_token:
            errors.append("INVITATION_ADMIN_TOKEN is required")

        if not self.cookie_secure:
            errors.append("COOKIE_SECURE must be true in production")

        if self.public_base_url.startswith("http://localhost"):
            errors.append("PUBLIC_BASE_URL must not be localhost in production")

        if self.storage_backend == "s3":
            if not self.s3_bucket:
                errors.append("S3_BUCKET is required when STORAGE_BACKEND=s3")
            if not self.s3_access_key_id:
                errors.append("S3_ACCESS_KEY_ID is required when STORAGE_BACKEND=s3")
            if not self.s3_secret_access_key:
                errors.append("S3_SECRET_ACCESS_KEY is required when STORAGE_BACKEND=s3")

        if errors:
            raise ValidationError.from_exception_data(
                title="ProductionConfigurationError",
                line_errors=[
                    {"type": "value_error", "msg": e, "loc": ("production",)} for e in errors
                ],
            )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_settings_uncached() -> Settings:
    """Get uncached settings instance (for testing)."""
    return Settings()


def clear_settings_cache() -> None:
    """Clear the settings cache (for testing)."""
    get_settings.cache_clear()
