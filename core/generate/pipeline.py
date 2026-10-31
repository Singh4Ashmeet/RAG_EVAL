"""Context-grounded answer generation using existing retriever/provider contracts."""

from __future__ import annotations

import time
from collections.abc import Sequence

from core.errors import GenerationError
from core.generate.citations import resolve_citations
from core.generate.prompts import SYSTEM_PROMPT, build_grounded_user_prompt
from core.generate.schema import GeneratedAnswer
from core.llm.base import GenerationSettings, LLMProvider
from core.retrieval.base import Retriever
from core.retrieval.schema import SearchResult

NO_CONTEXT_ANSWER = (
    "I don't have enough information in the retrieved sources to answer that question."
)


class GroundedAnswerGenerator:
    """Retrieve evidence, generate a constrained answer, and retain its provenance."""

    def __init__(
        self,
        retriever: Retriever,
        provider: LLMProvider,
        *,
        top_k: int = 5,
        settings: GenerationSettings | None = None,
    ) -> None:
        self.retriever = retriever
        self.provider = provider
        self.top_k = self._validate_top_k(top_k)
        self.settings = settings or GenerationSettings()

    def answer(self, query: str, *, top_k: int | None = None) -> GeneratedAnswer:
        """Answer a query and return all retrieved chunks alongside the model response."""
        normalized_query = self._validate_query(query)
        effective_top_k = self.top_k if top_k is None else self._validate_top_k(top_k)
        started_at = time.perf_counter()
        sources = self.retriever.retrieve(normalized_query, top_k=effective_top_k)
        return self._answer_with_sources(normalized_query, sources, started_at)

    def answer_with_sources(
        self,
        query: str,
        sources: Sequence[SearchResult],
    ) -> GeneratedAnswer:
        """Generate from already retrieved results without issuing another retrieval.

        This supports evaluation runners that need to score one retrieval result
        and then generate from exactly that same context.
        """
        normalized_query = self._validate_query(query)
        started_at = time.perf_counter()
        return self._answer_with_sources(normalized_query, sources, started_at)

    def _answer_with_sources(
        self,
        normalized_query: str,
        sources: Sequence[SearchResult],
        started_at: float,
    ) -> GeneratedAnswer:
        if not sources:
            return GeneratedAnswer(
                query=normalized_query,
                answer=NO_CONTEXT_ANSWER,
                sources=[],
                citations=[],
                unresolved_citation_labels=[],
                model_id=self.provider.model_id,
                abstained=True,
                duration_ms=max(0.0, (time.perf_counter() - started_at) * 1000.0),
            )

        source_list = list(sources)
        user_prompt = build_grounded_user_prompt(normalized_query, source_list)
        try:
            answer_text = self.provider.generate(
                SYSTEM_PROMPT,
                user_prompt,
                settings=self.settings,
            )
        except Exception as exc:
            raise GenerationError(f"Answer generation failed: {exc}") from exc

        if not isinstance(answer_text, str) or not answer_text.strip():
            raise GenerationError("Language model returned an empty answer.")

        citations, unresolved = resolve_citations(answer_text, source_list)
        return GeneratedAnswer(
            query=normalized_query,
            answer=answer_text.strip(),
            sources=source_list,
            citations=citations,
            unresolved_citation_labels=unresolved,
            model_id=self.provider.model_id,
            abstained=False,
            duration_ms=max(0.0, (time.perf_counter() - started_at) * 1000.0),
        )

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise GenerationError("Query must be a non-empty string.")
        return query.strip()

    @staticmethod
    def _validate_top_k(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise GenerationError("top_k must be a positive integer.")
        return value
