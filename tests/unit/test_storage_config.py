"""Tests for production-safe S3-compatible storage configuration.

Validates:
- Production startup rejects missing or invalid S3 configuration.
- Development and test modes accept local storage without requiring S3 config.
- AWS-prefixed and legacy S3-prefixed environment variables resolve correctly.
- Production detection is consistent between APP_ENV=production and RENDER=true.
- API never silently falls back to local disk in production.
- API and worker resolve the same storage configuration.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError


def _clear_storage_env(monkeypatch):
    """Remove all S3/storage-related environment variables."""
    for key in [
        "STORAGE_BACKEND",
        "S3_BUCKET",
        "AWS_BUCKET",
        "S3_ENDPOINT_URL",
        "AWS_ENDPOINT_URL",
        "S3_REGION",
        "AWS_REGION",
        "S3_ACCESS_KEY_ID",
        "AWS_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "S3_PREFIX",
        "APP_ENV",
        "RENDER",
    ]:
        monkeypatch.delenv(key, raising=False)


def _make_settings(**overrides):
    """Create an isolated Settings instance with the given overrides."""
    from core.config import Settings

    return Settings(**overrides)


def _valid_prod_kwargs(**extras):
    """Return valid baseline settings for production instantiation."""
    base = {
        "app_env": "production",
        "storage_backend": "s3",
        "s3_bucket": "rag-eval-private",
        "s3_endpoint_url": "https://xyz.supabase.co/storage/v1/s3",
        "s3_region": "ap-south-1",
        "s3_access_key_id": "test-key-id",
        "s3_secret_access_key": "test-secret-key",
        "cookie_secure": True,
        "public_base_url": "https://app.example.com",
        "credential_encryption_key": "x" * 44,
        "invitation_admin_token": "admin-tok",
        "database_url": "postgresql://u:p@host/db",
    }
    base.update(extras)
    return base


# =========================================================================
# 1. Production startup rejects missing S3 configuration
# =========================================================================
class TestProductionRejectsMissing:
    def test_production_rejects_missing_s3_bucket(self, monkeypatch):
        _clear_storage_env(monkeypatch)
        kwargs = _valid_prod_kwargs()
        kwargs["s3_bucket"] = ""
        with pytest.raises(ValidationError, match="s3_bucket"):
            _make_settings(**kwargs)

    def test_production_rejects_missing_s3_endpoint(self, monkeypatch):
        _clear_storage_env(monkeypatch)
        kwargs = _valid_prod_kwargs()
        kwargs["s3_endpoint_url"] = ""
        with pytest.raises(ValidationError, match="s3_endpoint_url"):
            _make_settings(**kwargs)

    def test_production_rejects_missing_access_key(self, monkeypatch):
        _clear_storage_env(monkeypatch)
        kwargs = _valid_prod_kwargs()
        kwargs["s3_access_key_id"] = ""
        with pytest.raises(ValidationError, match="s3_access_key_id"):
            _make_settings(**kwargs)

    def test_production_rejects_missing_secret_key(self, monkeypatch):
        _clear_storage_env(monkeypatch)
        kwargs = _valid_prod_kwargs()
        kwargs["s3_secret_access_key"] = ""
        with pytest.raises(ValidationError, match="s3_secret_access_key"):
            _make_settings(**kwargs)

    def test_production_rejects_missing_s3_region(self, monkeypatch):
        """Task 2: Production must reject missing S3 region without implicit default."""
        _clear_storage_env(monkeypatch)
        kwargs = _valid_prod_kwargs()
        kwargs["s3_region"] = ""
        with pytest.raises(ValidationError, match="s3_region"):
            _make_settings(**kwargs)

    def test_production_rejects_http_endpoint(self, monkeypatch):
        _clear_storage_env(monkeypatch)
        kwargs = _valid_prod_kwargs(s3_endpoint_url="http://insecure.example.com/storage")
        with pytest.raises(ValidationError, match="HTTPS"):
            _make_settings(**kwargs)

    def test_production_rejects_local_storage_backend(self, monkeypatch):
        _clear_storage_env(monkeypatch)
        kwargs = _valid_prod_kwargs(storage_backend="local")
        with pytest.raises(ValidationError, match="STORAGE_BACKEND"):
            _make_settings(**kwargs)


# =========================================================================
# 2. Region and Aliases
# =========================================================================
class TestRegionAndAliases:
    def test_explicit_aws_region_accepted_in_production(self, monkeypatch):
        """Task 2: Explicit AWS_REGION is accepted in production."""
        _clear_storage_env(monkeypatch)
        monkeypatch.setenv("AWS_REGION", "ap-south-1")
        settings = _make_settings(**_valid_prod_kwargs(s3_region="ap-south-1"))
        assert settings.s3_region == "ap-south-1"

    def test_explicit_s3_region_accepted_in_production(self, monkeypatch):
        """Task 2: Explicit S3_REGION is accepted in production."""
        _clear_storage_env(monkeypatch)
        monkeypatch.setenv("S3_REGION", "eu-west-1")
        settings = _make_settings(**_valid_prod_kwargs(s3_region="eu-west-1"))
        assert settings.s3_region == "eu-west-1"

    def test_development_valid_without_region(self, monkeypatch):
        """Task 2: Development does not require a region."""
        _clear_storage_env(monkeypatch)
        settings = _make_settings(app_env="development", storage_backend="local")
        assert settings.s3_region == ""
        assert settings.storage_backend == "local"

    def test_aws_prefixed_aliases_resolve(self, monkeypatch):
        _clear_storage_env(monkeypatch)
        monkeypatch.setenv("AWS_REGION", "ap-south-1")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret123")
        monkeypatch.setenv("AWS_BUCKET", "aws-bucket-test")
        monkeypatch.setenv("AWS_ENDPOINT_URL", "https://aws.endpoint.test")
        settings = _make_settings(app_env="development")
        assert settings.s3_region == "ap-south-1"
        assert settings.s3_access_key_id == "AKIATEST"
        assert settings.s3_secret_access_key == "secret123"
        assert settings.s3_bucket == "aws-bucket-test"
        assert settings.s3_endpoint_url == "https://aws.endpoint.test"

    def test_legacy_s3_aliases_resolve(self, monkeypatch):
        _clear_storage_env(monkeypatch)
        monkeypatch.setenv("S3_REGION", "eu-central-1")
        monkeypatch.setenv("S3_ACCESS_KEY_ID", "legacy-key")
        monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "legacy-secret")
        monkeypatch.setenv("S3_BUCKET", "s3-bucket-test")
        monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.endpoint.test")
        settings = _make_settings(app_env="development")
        assert settings.s3_region == "eu-central-1"
        assert settings.s3_access_key_id == "legacy-key"
        assert settings.s3_secret_access_key == "legacy-secret"
        assert settings.s3_bucket == "s3-bucket-test"
        assert settings.s3_endpoint_url == "https://s3.endpoint.test"

    def test_aws_prefix_precedence_over_s3_prefix(self, monkeypatch):
        _clear_storage_env(monkeypatch)
        monkeypatch.setenv("AWS_REGION", "us-west-2")
        monkeypatch.setenv("S3_REGION", "eu-central-1")
        settings = _make_settings(app_env="development")
        assert settings.s3_region == "us-west-2"


# =========================================================================
# 3. Production Detection Consistency (Task 4)
# =========================================================================
class TestProductionDetection:
    def test_app_env_production_triggers_is_production(self, monkeypatch):
        _clear_storage_env(monkeypatch)
        settings = _make_settings(**_valid_prod_kwargs())
        assert settings.is_production is True
        assert settings.is_development is False

    def test_render_env_triggers_is_production(self, monkeypatch):
        """Task 4: RENDER=true causes is_production to be True."""
        _clear_storage_env(monkeypatch)
        monkeypatch.setenv("RENDER", "true")
        kwargs = _valid_prod_kwargs(app_env="development")
        settings = _make_settings(**kwargs)
        assert settings.is_production is True

    def test_ordinary_development_is_not_production(self, monkeypatch):
        """Task 4: Ordinary local development is not classified as production."""
        _clear_storage_env(monkeypatch)
        monkeypatch.delenv("RENDER", raising=False)
        settings = _make_settings(app_env="development", storage_backend="local")
        assert settings.is_production is False
        assert settings.is_development is True

    def test_test_mode_is_not_production(self, monkeypatch):
        _clear_storage_env(monkeypatch)
        monkeypatch.delenv("RENDER", raising=False)
        settings = _make_settings(app_env="test", storage_backend="local")
        assert settings.is_production is False
        assert settings.is_test is True


# =========================================================================
# 4. Storage Behavior & No Fallback (Task 3)
# =========================================================================
class TestStorageBehavior:
    def test_store_original_raises_in_production_without_s3(self, monkeypatch):
        """Task 3: Production without S3 raises StorageConfigurationError without class mutation."""
        _clear_storage_env(monkeypatch)
        from apps.api.main import StorageConfigurationError, store_original
        from core.config import clear_settings_cache

        clear_settings_cache()
        mock_cfg = MagicMock()
        mock_cfg.is_production = True
        mock_cfg.storage_backend = "local"
        mock_cfg.s3_bucket = ""

        with patch("apps.api.main.get_settings", return_value=mock_cfg):
            with pytest.raises(StorageConfigurationError, match="not permitted in production"):
                store_original("test/doc.txt", b"data", "text/plain")

        clear_settings_cache()

    def test_delete_original_raises_in_production_without_s3(self, monkeypatch):
        """Task 3: delete_original in production without S3 raises StorageConfigurationError."""
        _clear_storage_env(monkeypatch)
        from apps.api.main import StorageConfigurationError, delete_original
        from core.config import clear_settings_cache

        clear_settings_cache()
        mock_cfg = MagicMock()
        mock_cfg.is_production = True
        mock_cfg.storage_backend = "local"
        mock_cfg.s3_bucket = ""

        with patch("apps.api.main.get_settings", return_value=mock_cfg):
            with pytest.raises(StorageConfigurationError, match="not permitted in production"):
                delete_original("test/doc.txt")

        clear_settings_cache()

    def test_development_local_storage_stores_and_deletes(self, monkeypatch, tmp_path):
        """Task 3: Local storage works correctly in development."""
        _clear_storage_env(monkeypatch)
        from apps.api.main import delete_original, store_original
        from core.config import clear_settings_cache

        clear_settings_cache()
        monkeypatch.setattr("apps.api.main.PRIVATE_DATA", tmp_path)

        mock_cfg = MagicMock()
        mock_cfg.is_production = False
        mock_cfg.storage_backend = "local"
        mock_cfg.s3_bucket = ""

        with patch("apps.api.main.get_settings", return_value=mock_cfg):
            store_original("sub/doc.txt", b"hello world", "text/plain")
            saved_file = tmp_path / "sub" / "doc.txt"
            assert saved_file.is_file()
            assert saved_file.read_bytes() == b"hello world"

            delete_original("sub/doc.txt")
            assert not saved_file.exists()

        clear_settings_cache()

    def test_s3_storage_invoked_with_configured_options(self, monkeypatch):
        """Task 3: S3 storage passes explicit endpoint, region, and credentials."""
        _clear_storage_env(monkeypatch)
        import apps.api.main as api_mod
        from core.config import clear_settings_cache

        clear_settings_cache()
        mock_client = MagicMock()
        captured_kwargs = {}

        def mock_boto_client(service, **kwargs):
            captured_kwargs.update(kwargs)
            return mock_client

        monkeypatch.setattr("boto3.client", mock_boto_client)

        mock_cfg = MagicMock()
        mock_cfg.is_production = True
        mock_cfg.storage_backend = "s3"
        mock_cfg.s3_bucket = "my-test-bucket"
        mock_cfg.s3_endpoint_url = "https://xyz.supabase.co/storage/v1/s3"
        mock_cfg.s3_region = "ap-south-1"
        mock_cfg.s3_access_key_id = "key-123"
        mock_cfg.s3_secret_access_key = "secret-456"

        with patch("apps.api.main.get_settings", return_value=mock_cfg):
            api_mod.store_original("docs/item.txt", b"payload data", "text/plain")

            assert captured_kwargs.get("endpoint_url") == "https://xyz.supabase.co/storage/v1/s3"
            assert captured_kwargs.get("region_name") == "ap-south-1"
            assert captured_kwargs.get("aws_access_key_id") == "key-123"
            assert captured_kwargs.get("aws_secret_access_key") == "secret-456"

            # Custom Supabase endpoint avoids ServerSideEncryption argument
            mock_client.put_object.assert_called_once_with(
                Bucket="my-test-bucket",
                Key="docs/item.txt",
                Body=b"payload data",
                ContentType="text/plain",
            )

            # Test delete_original also calls S3 client
            api_mod.delete_original("docs/item.txt")
            mock_client.delete_object.assert_called_once_with(
                Bucket="my-test-bucket",
                Key="docs/item.txt",
            )

        clear_settings_cache()

    def test_s3_standard_aws_includes_sse(self, monkeypatch):
        """Task 3: Standard AWS S3 (no custom endpoint) includes ServerSideEncryption."""
        _clear_storage_env(monkeypatch)
        import apps.api.main as api_mod
        from core.config import clear_settings_cache

        clear_settings_cache()
        mock_client = MagicMock()
        monkeypatch.setattr("boto3.client", lambda *args, **kwargs: mock_client)

        mock_cfg = MagicMock()
        mock_cfg.is_production = True
        mock_cfg.storage_backend = "s3"
        mock_cfg.s3_bucket = "aws-bucket"
        mock_cfg.s3_endpoint_url = ""
        mock_cfg.s3_region = "us-east-1"
        mock_cfg.s3_access_key_id = "aws-key"
        mock_cfg.s3_secret_access_key = "aws-secret"

        with patch("apps.api.main.get_settings", return_value=mock_cfg):
            api_mod.store_original("docs/aws_item.txt", b"aws data", "text/plain")

            mock_client.put_object.assert_called_once_with(
                Bucket="aws-bucket",
                Key="docs/aws_item.txt",
                Body=b"aws data",
                ContentType="text/plain",
                ServerSideEncryption="AES256",
            )

        clear_settings_cache()


# =========================================================================
# 5. Worker and API Resolve Same Configuration
# =========================================================================
class TestWorkerAPIConsistency:
    def test_worker_and_api_use_same_settings(self, monkeypatch):
        _clear_storage_env(monkeypatch)
        monkeypatch.setenv("AWS_REGION", "ap-south-1")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secretexample")
        monkeypatch.setenv("S3_BUCKET", "rag-eval-private")
        monkeypatch.setenv("S3_ENDPOINT_URL", "https://xyz.supabase.co/storage/v1/s3")
        monkeypatch.setenv("STORAGE_BACKEND", "s3")

        from core.config import clear_settings_cache, get_settings_uncached

        clear_settings_cache()

        settings = get_settings_uncached()
        assert settings.s3_region == "ap-south-1"
        assert settings.s3_access_key_id == "AKIAEXAMPLE"
        assert settings.s3_secret_access_key == "secretexample"
        assert settings.s3_bucket == "rag-eval-private"
        assert settings.s3_endpoint_url == "https://xyz.supabase.co/storage/v1/s3"
        assert settings.storage_backend == "s3"

        clear_settings_cache()
