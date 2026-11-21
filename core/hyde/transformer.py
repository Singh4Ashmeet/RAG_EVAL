"""HyDE (Hypothetical Document Embeddings) transformer for query expansion."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from core.errors import RetrievalError
from core.llm.base import GenerationSettings
from core.llm.ollama_provider import OllamaLLMProvider

logger = logging.getLogger(__name__)


# Default HyDE prompt for generating hypothetical technical passages
_DEFAULT_HYDE_PROMPT = """Given the following user query:

{query}

Write a concise hypothetical technical passage that would likely appear in a documentation corpus containing the answer.

Requirements:
- Produce only the technical passage
- Do not mention that this is hypothetical
- Do not discuss the task or provide meta-commentary
- Write as if this passage exists in a real technical documentation corpus
- Include relevant technical terminology, specific values, and structural details
- Keep it concise (2-5 sentences)

Passage:"""


@dataclass(frozen=True)
class HyDEConfig:
    """Configuration for HyDE generation."""

    model_name: str = "llama3.2:3b"
    ollama_host: str = "http://localhost:11434"
    timeout: float = 120.0
    temperature: float = 0.0
    max_tokens: int = 512
    prompt_template: str = _DEFAULT_HYDE_PROMPT
    max_passage_length: int = 2000
    min_passage_length: int = 50
    generation_timeout: float = 60.0

    def to_cache_key(self) -> str:
        """Generate a deterministic cache key for this configuration."""
        config_dict = {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "prompt_template": self.prompt_template,
            "max_passage_length": self.max_passage_length,
            "min_passage_length": self.min_passage_length,
        }
        serialized = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


class HyDEGenerationError(RetrievalError):
    """Raised when HyDE passage generation fails."""

    pass


class HyDETransformer:
    """
    HyDE (Hypothetical Document Embeddings) transformer.

    Generates a hypothetical technical passage from a user query that can be used
    as an alternative dense retrieval representation.
    """

    def __init__(
        self,
        config: HyDEConfig | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.config = config or HyDEConfig()
        self.cache_dir = cache_dir or Path.cwd() / ".hyde_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the LLM provider
        self._provider = OllamaLLMProvider(
            model=self.config.model_name,
            host=self.config.ollama_host,
            timeout=self.config.timeout,
        )
        self._settings = GenerationSettings(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # Compile cache key
        self._cache_key = self.config.to_cache_key()

    def transform(self, query: str) -> str:
        """
        Generate a hypothetical technical passage for the given query.

        Args:
            query: The original user query.

        Returns:
            A hypothetical technical passage suitable for dense retrieval.

        Raises:
            HyDEGenerationError: If generation fails or output is invalid.
        """
        if not query or not query.strip():
            raise HyDEGenerationError("Query cannot be empty")

        query = query.strip()

        # Check cache first
        cache_hit, cached_passage = self._check_cache(query)
        if cache_hit and cached_passage is not None:
            logger.debug(f"HyDE cache hit for query: {query[:50]}...")
            return cached_passage

        # Generate hypothetical passage
        start_time = time.perf_counter()
        passage = self._generate_passage(query)
        gen_time = time.perf_counter() - start_time

        logger.info(f"HyDE generated passage in {gen_time:.3f}s for query: {query[:50]}...")

        # Cache the result
        self._save_cache(query, passage)

        return passage

    def _generate_passage(self, query: str) -> str:
        """Generate a hypothetical passage using the LLM."""
        prompt_template = self.config.prompt_template or _DEFAULT_HYDE_PROMPT
        prompt = prompt_template.format(query=query)

        try:
            response = self._provider.generate(
                system_prompt="You are a technical documentation expert.",
                user_prompt=prompt,
                settings=self._settings,
            )

            passage = response.strip()

            # Validate the generated passage
            validated = self._validate_passage(passage)
            return validated

        except Exception as exc:
            raise HyDEGenerationError(f"HyDE generation failed: {exc}") from exc

    def _validate_passage(self, passage: str) -> str:
        """Validate and clean the generated passage."""
        if not passage:
            raise HyDEGenerationError("Generated passage is empty")

        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", passage).strip()

        if not cleaned:
            raise HyDEGenerationError("Generated passage is empty after cleaning")

        if len(cleaned) < self.config.min_passage_length:
            raise HyDEGenerationError(
                f"Generated passage too short ({len(cleaned)} chars), "
                f"minimum is {self.config.min_passage_length}"
            )

        if len(cleaned) > self.config.max_passage_length:
            # Truncate if too long
            cleaned = cleaned[: self.config.max_passage_length].rsplit(" ", 1)[0] + "..."

        return cleaned

    def _check_cache(self, query: str) -> tuple[bool, str | None]:
        """Check if a cached result exists for the query."""
        cache_key = hashlib.sha256(f"{self._cache_key}:{query}".encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return False, None

        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if data.get("query") == query and data.get("config_key") == self._cache_key:
                return True, data.get("passage")
        except Exception:
            pass

        return False, None

    def _save_cache(self, query: str, passage: str) -> None:
        """Save the generated passage to cache."""
        cache_key = hashlib.sha256(f"{self._cache_key}:{query}".encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            data = {
                "query": query,
                "passage": passage,
                "config_key": self._cache_key,
                "timestamp": time.time(),
            }
            cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Failed to save HyDE cache: {exc}")

    def clear_cache(self) -> int:
        """Clear all cached HyDE results. Returns number of files removed."""
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except Exception:
                pass
        return count
