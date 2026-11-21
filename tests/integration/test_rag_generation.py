"""Composed retrieval and answer generation test without external services."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.embed.ollama_embedder import OllamaEmbedder
from core.generate import GroundedAnswerGenerator
from core.ingest import create_default_ingestion_pipeline
from core.llm import GenerationSettings
from core.llm.ollama_provider import OllamaLLMProvider
from core.normalize.schema import Chunk
from core.retrieval import DenseRetriever
from core.storage import SQLiteVectorStore


class FixedEmbedder:
    model_id = "test:fixed-v1"

    @property
    def dimension(self) -> int:
        return 2

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class EvidenceEchoProvider:
    model_id = "test:answer-v1"

    def generate(self, system_prompt, user_prompt, *, settings: GenerationSettings) -> str:
        assert "only the retrieved source" in system_prompt
        assert '"label":"S1"' in user_prompt
        return "The document says the local workflow is reproducible [S1]."


def test_local_retrieval_to_grounded_answer_flow(tmp_path: Path) -> None:
    chunk = Chunk(
        chunk_id="doc-a_c0000",
        document_id="doc-a",
        text="The local workflow is reproducible.",
        chunk_index=0,
        source_element_ids=["doc-a_e0000"],
    )
    with SQLiteVectorStore(tmp_path / "rag.sqlite3") as store:
        retriever = DenseRetriever(FixedEmbedder(), store)
        retriever.index_document("doc-a", [chunk])
        answerer = GroundedAnswerGenerator(
            retriever,
            EvidenceEchoProvider(),
            settings=GenerationSettings(temperature=0.0, max_tokens=64),
        )

        result = answerer.answer("What does the document say?")

    assert result.answer == "The document says the local workflow is reproducible [S1]."
    assert result.abstained is False
    assert result.model_id == "test:answer-v1"
    assert [(citation.label, citation.chunk_id) for citation in result.citations] == [
        ("S1", "doc-a_c0000")
    ]
    assert result.sources[0].chunk.source_element_ids == ["doc-a_e0000"]


@pytest.mark.live
def test_live_ollama_retrieval_and_answer_generation(tmp_path: Path) -> None:
    """Live end-to-end RAG verification using real Ollama embedder and generator."""
    sample_file = tmp_path / "system_architecture.md"
    sample_file.write_text(
        "# System Architecture\n\n"
        "The evaluation engine utilizes an SQLite vector store for local execution.\n"
        "Deterministic chunking is performed using ElementAwareChunker.\n",
        encoding="utf-8",
    )

    pipeline = create_default_ingestion_pipeline(chunk_size=200, chunk_overlap=20)
    ingest_result = pipeline.ingest(sample_file)
    assert ingest_result.document is not None
    assert len(ingest_result.chunks) > 0

    embedder = OllamaEmbedder(model="nomic-embed-text")
    llm = OllamaLLMProvider(model="llama3.2:3b")

    with SQLiteVectorStore(tmp_path / "live_rag.sqlite3") as store:
        retriever = DenseRetriever(embedder, store)
        doc_id = ingest_result.document.metadata.document_id
        retriever.index_document(doc_id, ingest_result.chunks)
        assert store.count() > 0

        matches = retriever.retrieve("What does the evaluation engine utilize?", top_k=2)
        assert len(matches) > 0

        generator = GroundedAnswerGenerator(
            retriever,
            llm,
            settings=GenerationSettings(temperature=0.0, max_tokens=100),
        )
        answer_result = generator.answer(
            "What does the evaluation engine utilize for local execution?"
        )

        assert answer_result.abstained is False
        assert len(answer_result.answer.strip()) > 0
        assert len(answer_result.citations) > 0
        assert answer_result.citations[0].label == "S1"
