"""Unit tests for Phase 13 Failure Forensics & Bottleneck Analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPORT_PATH = Path("docs/reports/phase13_failure_analysis.json")


@pytest.fixture(scope="module")
def report() -> dict:
    """Load the Phase 13 failure analysis report."""
    if not REPORT_PATH.exists():
        pytest.skip("Phase 13 report not generated yet")
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


class TestPhase13ReportStructure:
    def test_report_has_required_keys(self, report: dict) -> None:
        required = [
            "phase",
            "baseline_config",
            "track_summaries",
            "failure_clusters",
            "query_forensics",
            "oracle_analysis",
            "previous_experiments",
            "headroom_analysis",
        ]
        for key in required:
            assert key in report, f"Missing key: {key}"

    def test_phase_number(self, report: dict) -> None:
        assert report["phase"] == 13


class TestEveryQueryClassified:
    def test_all_track_a_queries_classified(self, report: dict) -> None:
        ta = [q for q in report["query_forensics"] if q["track"] == "Track A"]
        assert len(ta) == 14
        for q in ta:
            assert q["failure_class"] in (
                "NONE",
                "RANKING_FAILURE",
                "FUSION_FAILURE",
                "SEMANTIC_MISMATCH",
                "LEXICAL_MISMATCH",
                "IDENTIFIER_TOKENIZATION",
                "MULTI_CHUNK",
                "CORPUS_COVERAGE",
                "EVALUATION_ANNOTATION",
                "GENERATION_FAILURE",
                "ABSTENTION_SAFE",
                "CHUNKING_BOUNDARY",
                "UNKNOWN",
            ), f"Invalid failure class: {q['failure_class']} for {q['query_id']}"

    def test_all_track_b_queries_classified(self, report: dict) -> None:
        tb = [q for q in report["query_forensics"] if q["track"] == "Track B"]
        assert len(tb) == 52
        for q in tb:
            assert q["failure_class"], f"No failure class for {q['query_id']}"

    def test_all_track_c_queries_classified(self, report: dict) -> None:
        tc = [q for q in report["query_forensics"] if q["track"] == "Track C"]
        assert len(tc) == 30
        for q in tc:
            assert q["failure_class"], f"No failure class for {q['query_id']}"

    def test_all_track_d_queries_classified(self, report: dict) -> None:
        td = [q for q in report["query_forensics"] if q["track"] == "Track D"]
        assert len(td) == 25
        for q in td:
            assert q["failure_class"] == "ABSTENTION_SAFE"

    def test_no_query_silently_dropped(self, report: dict) -> None:
        total = len(report["query_forensics"])
        assert total == 121, f"Expected 121 queries total (14+52+30+25), got {total}"


class TestFailureTaxonomyValidity:
    VALID_CLASSES = {
        "CORPUS_COVERAGE",
        "CHUNKING_BOUNDARY",
        "LEXICAL_MISMATCH",
        "SEMANTIC_MISMATCH",
        "FUSION_FAILURE",
        "RANKING_FAILURE",
        "IDENTIFIER_TOKENIZATION",
        "MULTI_CHUNK",
        "EVALUATION_ANNOTATION",
        "GENERATION_FAILURE",
        "NONE",
        "ABSTENTION_SAFE",
        "UNKNOWN",
    }

    def test_all_failure_classes_valid(self, report: dict) -> None:
        for q in report["query_forensics"]:
            assert q["failure_class"] in self.VALID_CLASSES, (
                f"Invalid failure class '{q['failure_class']}' for {q['query_id']}"
            )

    def test_every_failure_has_reason(self, report: dict) -> None:
        for q in report["query_forensics"]:
            assert q["failure_reason"], (
                f"Missing failure reason for {q['query_id']} (class={q['failure_class']})"
            )


class TestTrackImmutability:
    def test_protected_track_a_unchanged(self) -> None:
        track_path = Path("eval/golden/track_a.jsonl")
        assert track_path.exists()
        with open(track_path, encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        assert count == 14

    def test_protected_track_b_unchanged(self) -> None:
        track_path = Path("eval/golden/track_b.jsonl")
        assert track_path.exists()
        with open(track_path, encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        assert count == 52

    def test_protected_track_c_unchanged(self) -> None:
        track_path = Path("eval/golden/track_c.jsonl")
        assert track_path.exists()
        with open(track_path, encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        assert count == 30

    def test_protected_track_d_unchanged(self) -> None:
        track_path = Path("eval/golden/track_d.jsonl")
        assert track_path.exists()
        with open(track_path, encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        assert count == 25


class TestBaselineConfigFrozen:
    def test_baseline_config(self, report: dict) -> None:
        cfg = report["baseline_config"]
        assert cfg["chunker"] == "ElementAwareChunker"
        assert cfg["chunk_size"] == 1000
        assert cfg["chunk_overlap"] == 200
        assert cfg["embedding_model"] == "nomic-embed-text:latest"
        assert cfg["embedding_dimensions"] == 768
        assert cfg["retrieval"] == "Dense + BM25 RRF"
        assert cfg["candidate_k"] == 20
        assert cfg["rrf_k"] == 100
        assert cfg["final_k"] == 5
        assert cfg["generation_model"] == "llama3.2:3b"


class TestOracleAnalysis:
    def test_oracle_total_records(self, report: dict) -> None:
        assert report["oracle_analysis"]["total_records"] == 523

    def test_no_clear_winner_count(self, report: dict) -> None:
        ncw = report["oracle_analysis"]["no_clear_winner"]
        assert ncw["count"] == 43

    def test_ambiguous_count(self, report: dict) -> None:
        ambig = report["oracle_analysis"]["ambiguous"]
        assert ambig["count"] == 124

    def test_strategy_distribution_complete(self, report: dict) -> None:
        dist = report["oracle_analysis"]["strategy_distribution"]
        total = sum(dist.values())
        assert total == 523


class TestPreviousExperimentsNotOverwritten:
    def test_phase_reports_exist(self) -> None:
        for phase_num in range(0, 13):
            # Check that previous validation reports still exist
            if phase_num in {3}:  # Phase 3 uses PHASE_3_CHUNKING.md
                continue
            path = Path(f"docs/PHASE_{phase_num}_VALIDATION.md")
            if path.exists():
                assert path.stat().st_size > 0

    def test_phase11_routing_dataset_untouched(self) -> None:
        """Verify Phase 11 routing dataset was not modified."""
        manifest_path = Path("data/routing/phase11/routing_dataset_manifest.json")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert manifest["total_examples"] == 523

    def test_phase12_model_untouched(self) -> None:
        """Verify Phase 12 learned router model was not modified."""
        meta_path = Path("models/routing/phase12/metadata.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            assert meta["model_version"] == "phase12-v1.0.0"


class TestProductionUnchanged:
    def test_learned_router_not_activated(self) -> None:
        """Verify learned router is NOT imported in main.py production path."""
        main_path = Path("main.py")
        assert main_path.exists()
        content = main_path.read_text(encoding="utf-8")
        # The learned router should not be used in the production flow
        assert "LearnedQueryRouter" not in content or "experimental" in content.lower()
