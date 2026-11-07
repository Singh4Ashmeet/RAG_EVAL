"""Tests for GenerationResponse data structure."""

from __future__ import annotations

import pytest

from core.llm import GenerationResponse


class TestGenerationResponse:
    """Tests for the GenerationResponse dataclass."""

    def test_construct_with_text_only(self) -> None:
        """Test constructing GenerationResponse with only text."""
        response = GenerationResponse(text="Hello, world!")
        assert response.text == "Hello, world!"
        assert response.input_tokens is None
        assert response.output_tokens is None
        assert response.total_tokens is None
        assert response.provider_metadata == {}

    def test_construct_with_all_token_fields(self) -> None:
        """Test constructing GenerationResponse with all token fields."""
        response = GenerationResponse(
            text="Hello, world!",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
        assert response.text == "Hello, world!"
        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.total_tokens == 150
        assert response.provider_metadata == {}

    def test_missing_token_fields_remain_none(self) -> None:
        """Test that missing token fields default to None."""
        response = GenerationResponse(text="Test", input_tokens=10)
        assert response.input_tokens == 10
        assert response.output_tokens is None
        assert response.total_tokens is None

        response = GenerationResponse(text="Test", output_tokens=5)
        assert response.input_tokens is None
        assert response.output_tokens == 5
        assert response.total_tokens is None

        response = GenerationResponse(text="Test", total_tokens=15)
        assert response.input_tokens is None
        assert response.output_tokens is None
        assert response.total_tokens == 15

    def test_explicit_zero_token_values_remain_zero(self) -> None:
        """Test that explicit zero token values are preserved."""
        response = GenerationResponse(
            text="Test",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
        )
        assert response.input_tokens == 0
        assert response.output_tokens == 0
        assert response.total_tokens == 0

    def test_partial_token_metadata_supported(self) -> None:
        """Test that partial token metadata is supported."""
        # Only input tokens
        r1 = GenerationResponse(text="Test", input_tokens=42)
        assert r1.input_tokens == 42
        assert r1.output_tokens is None
        assert r1.total_tokens is None

        # Only output tokens
        r2 = GenerationResponse(text="Test", output_tokens=21)
        assert r2.input_tokens is None
        assert r2.output_tokens == 21
        assert r2.total_tokens is None

        # Input and output, no total -> total is computed automatically
        r3 = GenerationResponse(text="Test", input_tokens=10, output_tokens=5)
        assert r3.input_tokens == 10
        assert r3.output_tokens == 5
        assert r3.total_tokens == 15

    def test_provider_metadata_defaults_to_independent_empty_dict(self) -> None:
        """Test that provider_metadata defaults to an independent empty dict."""
        r1 = GenerationResponse(text="Test")
        r2 = GenerationResponse(text="Test")

        # Each instance gets its own dict (not shared)
        assert r1.provider_metadata is not r2.provider_metadata
        assert r1.provider_metadata == {}
        assert r2.provider_metadata == {}

        # Modifying one doesn't affect the other
        r1.provider_metadata["key"] = "value"
        assert r2.provider_metadata == {}

    def test_provider_metadata_can_contain_additional_fields(self) -> None:
        """Test that provider_metadata can hold harmless additional fields."""
        response = GenerationResponse(
            text="Test",
            provider_metadata={"model_version": "v1.2.3", "finish_reason": "stop"},
        )
        assert response.provider_metadata == {
            "model_version": "v1.2.3",
            "finish_reason": "stop",
        }

    def test_immutability_raises_on_attribute_reassignment(self) -> None:
        """Test that frozen dataclass prevents attribute reassignment."""
        response = GenerationResponse(text="Test")
        with pytest.raises(AttributeError):
            response.text = "Modified"

    def test_existing_llm_provider_protocol_unchanged(self) -> None:
        """Verify the existing LLMProvider protocol still expects str return."""
        from typing import get_type_hints

        from core.llm.base import LLMProvider

        hints = get_type_hints(LLMProvider.generate)
        # The return type should still be str
        assert hints["return"] is str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
