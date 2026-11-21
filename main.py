"""Main application entry point and production RAG pipeline orchestrator.

Connects the validated production retrieval and generation configuration:
- Ingestion: PlainTextParser, Markdown, PDFParser, DOCXParser
- Normalization: Canonical NormalizedDocument representation
- Chunking: ElementAwareChunker (chunk_size=1000, chunk_overlap=200)
- Embedding: nomic-embed-text:latest (768 dimensions)
- Storage: SQLiteVectorStore (persistent vector storage)
- Lexical: BM25Retriever with technical term preservation
- Dense: DenseRetriever with cosine similarity
- Fusion: HybridRetriever (Dense + BM25 RRF, candidate_k=20, rrf_k=100, final_k=5)
- Generation: GroundedAnswerGenerator with llama3.2:3b (evidence-grounded)
- Citations & Provenance: Source element and chunk provenance tracking
- Abstention: Track D deterministic grounded refusal preservation
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import dotenv

from core.config import Settings, get_settings
from core.embed.base import Embedder
from core.embed.ollama_embedder import OllamaEmbedder
from core.generate.pipeline import GroundedAnswerGenerator
from core.generate.schema import GeneratedAnswer
from core.ingest.pipeline import IngestionPipeline, create_default_ingestion_pipeline
from core.ingest.schema import IngestionResult
from core.llm.base import GenerationSettings, LLMProvider
from core.llm.ollama_provider import OllamaLLMProvider
from core.normalize.schema import Chunk
from core.retrieval.bm25 import BM25Retriever
from core.retrieval.dense import DenseRetriever
from core.retrieval.hybrid import HybridRetriever
from core.retrieval.schema import SearchResult
from core.storage.base import VectorStore
from core.storage.sqlite_store import SQLiteVectorStore

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"

dotenv.load_dotenv(ROOT / ".env")


class ProductionRAGPipeline:
    """Production RAG pipeline connecting ingestion, chunking, embedding,

    storage, hybrid retrieval (Dense + BM25 RRF), generation, and citations.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        llm_provider: LLMProvider | None = None,
        vector_store: VectorStore | None = None,
        db_path: Path | str | None = None,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        candidate_k: int = 20,
        rrf_k: int = 100,
        final_k: int = 5,
    ) -> None:
        self.settings = settings or get_settings()
        self.chunk_size = chunk_size or self.settings.chunk_size
        self.chunk_overlap = chunk_overlap or self.settings.chunk_overlap
        self.candidate_k = candidate_k or self.settings.retrieval_candidate_k
        self.rrf_k = rrf_k or self.settings.retrieval_rrf_k
        self.final_k = final_k or self.settings.retrieval_final_k

        # 1. Ingestion & Chunking
        self.ingestion_pipeline: IngestionPipeline = create_default_ingestion_pipeline(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # 2. Embedding Model (Production default: nomic-embed-text)
        self.embedder: Embedder = embedder or OllamaEmbedder(
            model=self.settings.embedding_model,
            host=self.settings.embedding_base_url,
        )

        # 3. Vector Storage
        self._db_path = db_path or (ROOT / "data" / "production_rag.sqlite3")
        if vector_store is not None:
            self.vector_store: VectorStore = vector_store
        else:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self.vector_store = SQLiteVectorStore(self._db_path)

        # 4. Dense Retriever
        self.dense_retriever = DenseRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            model_id=self.settings.embedding_model,
        )

        # 5. In-memory chunk index for BM25
        self._indexed_chunks: dict[str, Chunk] = {}
        self.bm25_retriever: BM25Retriever | None = None

        # 6. Hybrid Retriever
        self.hybrid_retriever: HybridRetriever | None = None

        # 7. LLM Provider (Production default: llama3.2:3b)
        self.llm_provider: LLMProvider = llm_provider or OllamaLLMProvider(
            model=self.settings.chat_model,
            host=self.settings.chat_base_url,
        )

        # 8. Generation & Citation Engine
        self.answer_generator: GroundedAnswerGenerator | None = None
        self._rebuild_hybrid_retriever()

    def _rebuild_hybrid_retriever(self) -> None:
        """Reconstruct BM25 and HybridRetriever over currently indexed chunks."""
        chunks = list(self._indexed_chunks.values())
        if chunks:
            self.bm25_retriever = BM25Retriever(chunks)
            self.hybrid_retriever = HybridRetriever(
                dense_retriever=self.dense_retriever,
                bm25_retriever=self.bm25_retriever,
                fusion_method="rrf",
                rrf_k=self.rrf_k,
                candidate_k=self.candidate_k,
            )
        else:
            self.bm25_retriever = None
            self.hybrid_retriever = None

        if self.hybrid_retriever is not None:
            self.answer_generator = GroundedAnswerGenerator(
                retriever=self.hybrid_retriever,
                provider=self.llm_provider,
                top_k=self.final_k,
            )
        else:
            self.answer_generator = GroundedAnswerGenerator(
                retriever=self.dense_retriever,
                provider=self.llm_provider,
                top_k=self.final_k,
            )

    def ingest_file(self, file_path: Path | str) -> IngestionResult:
        """Ingest, normalize, chunk, and index a document from the filesystem."""
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        result = self.ingestion_pipeline.ingest(path)
        if result.status.value != "success" or not result.document:
            return result

        self.index_chunks(result.document.metadata.document_id, result.chunks)
        return result

    def index_chunks(self, document_id: str, chunks: Sequence[Chunk]) -> None:
        """Index chunks into both dense vector storage and BM25 index."""
        chunk_list = list(chunks)
        if not chunk_list:
            self.dense_retriever.delete_document(document_id)
            self._indexed_chunks = {
                cid: c for cid, c in self._indexed_chunks.items() if c.document_id != document_id
            }
            self._rebuild_hybrid_retriever()
            return

        self.dense_retriever.index_document(document_id, chunk_list)
        for chunk in chunk_list:
            self._indexed_chunks[chunk.chunk_id] = chunk
        self._rebuild_hybrid_retriever()

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Retrieve top chunks using production Dense + BM25 RRF fusion."""
        if not self._indexed_chunks:
            return []
        effective_k = min(top_k, self.final_k)
        if self.hybrid_retriever is not None:
            return self.hybrid_retriever.retrieve(query, top_k=effective_k)
        return self.dense_retriever.retrieve(query, top_k=effective_k)

    def answer(self, query: str, top_k: int = 5) -> GeneratedAnswer:
        """Retrieve evidence and generate an evidence-grounded answer with citations."""
        if not self._indexed_chunks:
            return GroundedAnswerGenerator(
                self.dense_retriever, self.llm_provider, top_k=top_k
            ).answer(query)
        effective_k = min(top_k, self.final_k)
        assert self.answer_generator is not None
        return self.answer_generator.answer(query, top_k=effective_k)

    def get_pipeline_info(self) -> dict[str, Any]:
        """Return production configuration and state."""
        return {
            "chunker": "ElementAwareChunker",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.settings.embedding_model,
            "chat_model": self.settings.chat_model,
            "retrieval_strategy": "Dense + BM25 RRF",
            "candidate_k": self.candidate_k,
            "rrf_k": self.rrf_k,
            "final_k": self.final_k,
            "indexed_chunks_count": len(self._indexed_chunks),
        }


def build_production_pipeline(
    settings: Settings | None = None,
    **kwargs: Any,
) -> ProductionRAGPipeline:
    """Factory creating a production-configured RAG pipeline."""
    return ProductionRAGPipeline(settings=settings, **kwargs)


def run_production_smoke_test() -> bool:
    """Execute a self-contained production end-to-end smoke test without external dependencies.

    Validates: Ingestion -> Chunking -> Embedding -> Storage -> BM25 -> Hybrid RRF -> Generation.
    """
    print("[SMOKE TEST] Initializing production RAG smoke test...", flush=True)

    class MockFixedEmbedder:
        model_id = "nomic-embed-text"
        dimension = 4

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0, 0.0]

    class MockEchoLLM:
        model_id = "llama3.2:3b"

        def generate(
            self,
            system_prompt: str,
            user_prompt: str,
            *,
            settings: GenerationSettings | None = None,
        ) -> str:
            assert "only the retrieved source" in system_prompt
            assert '"label":"S1"' in user_prompt or "S1" in user_prompt
            return "According to the production documentation, the RRF parameter k is 100 [S1]."

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        doc_path = tmp_path / "architecture.md"
        doc_path.write_text(
            "# Architecture Specification\n\n"
            "The production RAG pipeline uses ElementAwareChunker with 1000 chunk size.\n\n"
            "Hybrid retrieval combines Dense and BM25 using RRF with k=100 and candidate_k=20.\n",
            encoding="utf-8",
        )

        db_file = tmp_path / "smoke.sqlite3"
        empty_db_file = tmp_path / "empty.sqlite3"
        with (
            SQLiteVectorStore(db_file) as vector_store,
            SQLiteVectorStore(empty_db_file) as empty_vector_store,
        ):
            pipeline = ProductionRAGPipeline(
                embedder=MockFixedEmbedder(),  # type: ignore[arg-type]
                llm_provider=MockEchoLLM(),  # type: ignore[arg-type]
                vector_store=vector_store,
                chunk_size=1000,
                chunk_overlap=200,
                candidate_k=20,
                rrf_k=100,
                final_k=5,
            )

            # Ingestion & Chunking & Indexing
            ingest_res = pipeline.ingest_file(doc_path)
            assert ingest_res.status.value == "success", "Ingestion failed"
            assert ingest_res.document is not None, "Document failed to parse"
            assert len(ingest_res.chunks) >= 1, "No chunks generated"

            # Hybrid Retrieval
            retrieval_res = pipeline.retrieve("What is the RRF parameter?", top_k=5)
            assert len(retrieval_res) >= 1, "Hybrid retrieval returned empty results"
            assert retrieval_res[0].chunk.document_id == ingest_res.document.metadata.document_id

            # Grounded Generation & Citations
            answer_res = pipeline.answer("What is the RRF parameter?")
            assert answer_res.abstained is False, "Unexpected abstention on answerable query"
            assert "[S1]" in answer_res.answer, "Answer missing citation label"
            assert len(answer_res.citations) >= 1, "Citations not resolved"
            assert answer_res.citations[0].label == "S1"
            assert answer_res.model_id == "llama3.2:3b"

            # Abstention validation on unanswerable query when corpus is empty
            empty_pipeline = ProductionRAGPipeline(
                embedder=MockFixedEmbedder(),  # type: ignore[arg-type]
                llm_provider=MockEchoLLM(),  # type: ignore[arg-type]
                vector_store=empty_vector_store,
            )
            empty_answer = empty_pipeline.answer("Unanswerable question?")
            assert empty_answer.abstained is True, "Expected abstention on empty corpus"

    print("[SMOKE TEST] All production pipeline integration checks PASSED.", flush=True)
    return True


def _npm(*args: str, cwd: Path = FRONTEND) -> int:
    """Run npm on Windows (where npm is a .cmd file) and Unix."""
    command = ["npm", *args]
    return subprocess.call(command, cwd=cwd, shell=os.name == "nt")


def _install_frontend() -> None:
    vite_entry = FRONTEND / "node_modules" / "vite" / "bin" / "vite.js"
    if vite_entry.exists():
        return
    print("Installing frontend dependencies with npm ci...", flush=True)
    if _npm("ci") != 0:
        raise SystemExit("npm ci failed; check the Node.js and npm installation.")


def _commands() -> tuple[list[str], list[str]]:
    uv = shutil.which("uv")
    python = shutil.which("python") or sys.executable
    server = [
        "uvicorn",
        "apps.api.main:app",
        "--reload",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    backend = [uv, "run", *server] if uv else [python, "-m", *server]
    node = shutil.which("node")
    if not node:
        raise SystemExit("Node.js is required to run the frontend.")
    vite = str(FRONTEND / "node_modules" / "vite" / "bin" / "vite.js")
    frontend = [node, vite, "--host", "127.0.0.1", "--port", "5173"]
    return backend, frontend


def run_servers() -> int:
    if not (ROOT / "pyproject.toml").exists() or not (FRONTEND / "package.json").exists():
        raise SystemExit("Run main.py from a complete RAG_EVAL checkout.")
    if not shutil.which("uv") and not shutil.which("python"):
        raise SystemExit("Install uv or Python 3.12 before starting RAG_EVAL.")
    if not shutil.which("node") or not shutil.which("npm"):
        raise SystemExit("Install Node.js 20.19+ (or 22.12+) and npm before starting RAG_EVAL.")

    _install_frontend()
    backend_command, frontend_command = _commands()
    children: list[subprocess.Popen[bytes]] = []
    try:
        children.append(subprocess.Popen(backend_command, cwd=ROOT))
        children.append(subprocess.Popen(frontend_command, cwd=FRONTEND))
        print("RAG_EVAL is starting:", flush=True)
        print("  Frontend: http://127.0.0.1:5173", flush=True)
        print("  API docs: http://127.0.0.1:8000/docs", flush=True)
        while True:
            for child in children:
                code = child.poll()
                if code is not None:
                    return code
            time.sleep(0.25)
    except KeyboardInterrupt:
        return 0
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-eval",
        description="RAG_EVAL Production Orchestrator and Application Server",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the FastAPI backend and Vite frontend development servers.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run end-to-end production RAG pipeline smoke test.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a grounded query against the production RAG pipeline.",
    )
    parser.add_argument(
        "--ingest",
        type=str,
        default=None,
        help="Ingest a document file into the production pipeline.",
    )

    args = parser.parse_args(argv)

    if args.smoke_test:
        success = run_production_smoke_test()
        return 0 if success else 1

    if args.ingest:
        pipeline = build_production_pipeline()
        res = pipeline.ingest_file(args.ingest)
        print(f"Ingested {args.ingest}: status={res.status.value}, chunks={len(res.chunks)}")
        return 0 if res.status.value == "success" else 1

    if args.query:
        pipeline = build_production_pipeline()
        ans = pipeline.answer(args.query)
        print(f"Answer: {ans.answer}")
        for c in ans.citations:
            print(f"  Citation [{c.label}]: {c.chunk_id}")
        return 0

    # Default action: start both servers
    return run_servers()


if __name__ == "__main__":
    raise SystemExit(main())
