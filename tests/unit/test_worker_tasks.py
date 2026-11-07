"""Unit tests for worker task provider integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from apps.worker.tasks import evaluate_dataset, run_experiment


class TestWorkerTasksProviderIntegration:
    """Tests verifying worker tasks use the shared provider factory."""

    def _setup_common_mocks(self, mock_get_settings, mock_get_provider):
        """Set up common mock objects for tests."""
        mock_settings = MagicMock()
        mock_settings.chat_provider = "ollama"
        mock_get_settings.return_value = mock_settings

        mock_provider = MagicMock()
        mock_provider.model_id = "ollama:llama3.2:3b"
        mock_get_provider.return_value = mock_provider

        # Valid sha256 (64 chars) and valid cases list matching EvaluationCase schema
        valid_sha256 = "a" * 64
        valid_cases_json = (
            '[{"query_id": "q1", "query": "test query", "relevant_chunk_ids": ["chunk-1"]}]'
        )

        mock_revision = MagicMock()
        mock_revision.id = "rev-1"
        mock_revision.workspace_id = "ws-1"
        mock_revision.dataset_id = "ds-1"
        mock_revision.revision = 1
        mock_revision.sha256 = valid_sha256
        mock_revision.cases_json = valid_cases_json

        return mock_provider, mock_revision

    def test_evaluate_dataset_uses_get_chat_provider(self):
        """Test that evaluate_dataset obtains provider via get_chat_provider."""
        with (
            patch("apps.worker.tasks.get_chat_provider") as mock_get_provider,
            patch("eval.runners.evaluate.EvaluationRunner") as mock_runner,
            patch("apps.worker.tasks.HostedElementRetriever"),
            patch("apps.worker.tasks.GroundedAnswerGenerator"),
            patch("apps.worker.tasks.get_settings") as mock_get_settings,
        ):
            mock_provider, mock_revision = self._setup_common_mocks(
                mock_get_settings, mock_get_provider
            )

            mock_run = MagicMock()
            mock_run.id = "run-1"
            mock_run.workspace_id = "ws-1"
            mock_run.status = "pending"
            mock_run.report_json = None

            mock_job = MagicMock()
            mock_job.id = "job-1"
            mock_job.workspace_id = "ws-1"

            mock_report = MagicMock()
            mock_report.status = "complete"
            mock_report.queries = []
            mock_report.model_dump_json.return_value = '{"status": "complete", "queries": []}'
            mock_runner.return_value.run.return_value = mock_report

            mock_db = MagicMock()
            mock_db.bind.dialect.name = "sqlite"
            # Need enough values for all db.scalar calls including exception handler re-queries
            mock_db.scalar.side_effect = [
                mock_run,
                mock_job,
                mock_revision,
                [],  # First try
                mock_run,
                mock_job,  # After successful completion
                mock_run,
                mock_job,  # Exception handler (if any)
            ]
            mock_db.commit = MagicMock()
            mock_db.rollback = MagicMock()

            mock_session = MagicMock()
            mock_session.__enter__ = lambda self: mock_db
            mock_session.__exit__ = lambda self, *args: None

            with patch("apps.worker.tasks.SessionLocal", return_value=mock_session):
                try:
                    evaluate_dataset("ws-1", "run-1", "job-1", "rev-1", [1, 5, 10])
                except Exception:
                    pass  # We only care that get_chat_provider was called

            # Verify get_chat_provider was called with the chat_provider from settings
            mock_get_provider.assert_called_once_with("ollama")

    def test_run_experiment_uses_get_chat_provider(self):
        """Test that run_experiment obtains provider via get_chat_provider."""
        with (
            patch("apps.worker.tasks.get_chat_provider") as mock_get_provider,
            patch("eval.runners.experiment.ExperimentRunner") as mock_runner,
            patch("apps.worker.tasks.HostedElementRetriever"),
            patch("apps.worker.tasks.GroundedAnswerGenerator"),
            patch("apps.worker.tasks.get_settings") as mock_get_settings,
        ):
            mock_provider, mock_revision = self._setup_common_mocks(
                mock_get_settings, mock_get_provider
            )

            mock_experiment = MagicMock()
            mock_experiment.id = "exp-1"
            mock_experiment.workspace_id = "ws-1"
            mock_experiment.status = "pending"
            mock_experiment.report_json = None

            mock_job = MagicMock()
            mock_job.id = "job-1"
            mock_job.workspace_id = "ws-1"

            mock_report = MagicMock()
            mock_report.status = "complete"
            mock_report.variants = []
            mock_report.model_dump_json.return_value = '{"status": "complete", "variants": []}'
            mock_runner.return_value.run.return_value = mock_report

            mock_db = MagicMock()
            mock_db.bind.dialect.name = "sqlite"
            # Need enough values for all db.scalar calls including exception handler re-queries
            mock_db.scalar.side_effect = [
                mock_experiment,
                mock_job,
                mock_revision,
                [],  # First try
                mock_experiment,
                mock_job,  # After successful completion
                mock_experiment,
                mock_job,  # Exception handler (if any)
            ]
            mock_db.commit = MagicMock()
            mock_db.rollback = MagicMock()

            mock_session = MagicMock()
            mock_session.__enter__ = lambda self: mock_db
            mock_session.__exit__ = lambda self, *args: None

            payload = {
                "cutoffs": [1, 5, 10],
                "baseline_variant": "baseline",
                "bootstrap_resamples": 100,
                "random_seed": 42,
                "variants": [
                    {"name": "variant1", "retrieval_depth": 5},
                    {"name": "variant2", "retrieval_depth": 10},
                ],
            }

            with patch("apps.worker.tasks.SessionLocal", return_value=mock_session):
                try:
                    run_experiment("ws-1", "exp-1", "job-1", "rev-1", payload)
                except Exception:
                    pass  # We only care that get_chat_provider was called

            # Verify get_chat_provider was called for each variant (2 variants)
            assert mock_get_provider.call_count == 2
            for call in mock_get_provider.call_args_list:
                assert call.args == ("ollama",)

    def test_evaluate_dataset_passes_provider_to_grounded_answer_generator(self):
        """Test that evaluate_dataset passes the factory provider to GroundedAnswerGenerator."""
        with (
            patch("apps.worker.tasks.get_chat_provider") as mock_get_provider,
            patch("eval.runners.evaluate.EvaluationRunner"),
            patch("apps.worker.tasks.HostedElementRetriever"),
            patch("apps.worker.tasks.GroundedAnswerGenerator") as mock_generator_class,
            patch("apps.worker.tasks.get_settings") as mock_get_settings,
        ):
            mock_provider, mock_revision = self._setup_common_mocks(
                mock_get_settings, mock_get_provider
            )

            mock_run = MagicMock()
            mock_run.id = "run-1"
            mock_run.workspace_id = "ws-1"
            mock_run.status = "pending"

            mock_job = MagicMock()
            mock_job.id = "job-1"
            mock_job.workspace_id = "ws-1"

            mock_db = MagicMock()
            mock_db.bind.dialect.name = "sqlite"
            # Need enough values for all db.scalar calls including exception handler re-queries
            mock_db.scalar.side_effect = [
                mock_run,
                mock_job,
                mock_revision,
                [],  # First try
                mock_run,
                mock_job,  # After successful completion
                mock_run,
                mock_job,  # Exception handler (if any)
            ]
            mock_db.commit = MagicMock()

            mock_session = MagicMock()
            mock_session.__enter__ = lambda self: mock_db
            mock_session.__exit__ = lambda self, *args: None

            with patch("apps.worker.tasks.SessionLocal", return_value=mock_session):
                try:
                    evaluate_dataset("ws-1", "run-1", "job-1", "rev-1", [1, 5, 10])
                except Exception:
                    pass

            # Verify GroundedAnswerGenerator was called with the provider from get_chat_provider
            mock_generator_class.assert_called_once()
            args, kwargs = mock_generator_class.call_args
            assert args[1] is mock_provider  # Second positional arg is provider
            assert kwargs.get("top_k") == 10  # max_cutoff from [1, 5, 10]

    def test_run_experiment_passes_provider_to_grounded_answer_generator(self):
        """Test that run_experiment passes the factory provider to
        GroundedAnswerGenerator for each variant."""
        with (
            patch("apps.worker.tasks.get_chat_provider") as mock_get_provider,
            patch("eval.runners.experiment.ExperimentRunner"),
            patch("apps.worker.tasks.HostedElementRetriever"),
            patch("apps.worker.tasks.GroundedAnswerGenerator") as mock_generator_class,
            patch("apps.worker.tasks.get_settings") as mock_get_settings,
        ):
            mock_provider, mock_revision = self._setup_common_mocks(
                mock_get_settings, mock_get_provider
            )

            mock_experiment = MagicMock()
            mock_experiment.id = "exp-1"
            mock_experiment.workspace_id = "ws-1"
            mock_experiment.status = "pending"

            mock_job = MagicMock()
            mock_job.id = "job-1"
            mock_job.workspace_id = "ws-1"

            mock_db = MagicMock()
            mock_db.bind.dialect.name = "sqlite"
            # Need enough values for all db.scalar calls including exception handler re-queries
            mock_db.scalar.side_effect = [
                mock_experiment,
                mock_job,
                mock_revision,
                [],  # First try
                mock_experiment,
                mock_job,  # After successful completion
                mock_experiment,
                mock_job,  # Exception handler (if any)
            ]
            mock_db.commit = MagicMock()

            mock_session = MagicMock()
            mock_session.__enter__ = lambda self: mock_db
            mock_session.__exit__ = lambda self, *args: None

            payload = {
                "cutoffs": [1, 5, 10],
                "baseline_variant": "baseline",
                "bootstrap_resamples": 100,
                "random_seed": 42,
                "variants": [
                    {"name": "variant1", "retrieval_depth": 5},
                    {"name": "variant2", "retrieval_depth": 10},
                ],
            }

            with patch("apps.worker.tasks.SessionLocal", return_value=mock_session):
                try:
                    run_experiment("ws-1", "exp-1", "job-1", "rev-1", payload)
                except Exception:
                    pass

            # Verify GroundedAnswerGenerator was called for each variant with the provider
            assert mock_generator_class.call_count == 2
            for call in mock_generator_class.call_args_list:
                args, kwargs = call
                assert args[1] is mock_provider  # Second positional arg is provider
                assert kwargs.get("top_k") == 10  # max_cutoff from [1, 5, 10]

    def test_no_direct_ollama_llm_provider_import_in_tasks(self):
        """Test that tasks.py no longer imports OllamaLLMProvider directly."""
        import apps.worker.tasks as tasks_module

        # Verify the import is not present
        assert not hasattr(tasks_module, "OllamaLLMProvider")

    def test_provider_construction_failure_propagates(self):
        """Test that provider construction failures propagate through task error handling."""
        with (
            patch("apps.worker.tasks.get_chat_provider") as mock_get_provider,
            patch("eval.runners.evaluate.EvaluationRunner"),
            patch("apps.worker.tasks.HostedElementRetriever"),
            patch("apps.worker.tasks.GroundedAnswerGenerator"),
            patch("apps.worker.tasks.get_settings") as mock_get_settings,
        ):
            mock_settings = MagicMock()
            mock_settings.chat_provider = "ollama"
            mock_get_settings.return_value = mock_settings

            # Simulate provider construction failure
            mock_get_provider.side_effect = RuntimeError("Provider initialization failed")

            mock_run = MagicMock()
            mock_run.id = "run-1"
            mock_run.workspace_id = "ws-1"
            mock_run.status = "pending"

            mock_job = MagicMock()
            mock_job.id = "job-1"
            mock_job.workspace_id = "ws-1"

            valid_sha256 = "a" * 64
            valid_cases_json = (
                '[{"query_id": "q1", "query": "test query", "relevant_chunk_ids": ["chunk-1"]}]'
            )

            mock_revision = MagicMock()
            mock_revision.id = "rev-1"
            mock_revision.workspace_id = "ws-1"
            mock_revision.dataset_id = "ds-1"
            mock_revision.revision = 1
            mock_revision.sha256 = valid_sha256
            mock_revision.cases_json = valid_cases_json

            mock_db = MagicMock()
            mock_db.bind.dialect.name = "sqlite"
            # Order: run, job, revision, elements (first try), then run, job (in exception handler)
            mock_db.scalar.side_effect = [mock_run, mock_job, mock_revision, [], mock_run, mock_job]
            # Also mock scalars for the elements query
            mock_scalars_result = MagicMock()
            mock_scalars_result.__iter__ = lambda self: iter([])
            mock_db.scalars.return_value = mock_scalars_result
            mock_db.commit = MagicMock()
            mock_db.rollback = MagicMock()

            mock_session = MagicMock()
            mock_session.__enter__ = lambda self: mock_db
            mock_session.__exit__ = lambda self, *args: None

            with patch("apps.worker.tasks.SessionLocal", return_value=mock_session):
                with pytest.raises(RuntimeError, match="Provider initialization failed"):
                    evaluate_dataset("ws-1", "run-1", "job-1", "rev-1", [1, 5, 10])

            # Verify error handling: rollback was called
            mock_db.rollback.assert_called()

    def test_factory_not_wrapped_by_worker_retry_fallback(self):
        """Test that the factory provider is not wrapped a second time
        by worker-specific retry/fallback."""
        with (
            patch("apps.worker.tasks.get_chat_provider") as mock_get_provider,
            patch("eval.runners.evaluate.EvaluationRunner"),
            patch("apps.worker.tasks.HostedElementRetriever"),
            patch("apps.worker.tasks.GroundedAnswerGenerator") as mock_generator_class,
            patch("apps.worker.tasks.get_settings") as mock_get_settings,
        ):
            mock_provider, mock_revision = self._setup_common_mocks(
                mock_get_settings, mock_get_provider
            )

            mock_run = MagicMock()
            mock_run.id = "run-1"
            mock_run.workspace_id = "ws-1"
            mock_run.status = "pending"

            mock_job = MagicMock()
            mock_job.id = "job-1"
            mock_job.workspace_id = "ws-1"

            mock_db = MagicMock()
            mock_db.bind.dialect.name = "sqlite"
            # Need enough values for all db.scalar calls including exception handler re-queries
            mock_db.scalar.side_effect = [
                mock_run,
                mock_job,
                mock_revision,
                [],  # First try
                mock_run,
                mock_job,  # After successful completion
                mock_run,
                mock_job,  # Exception handler (if any)
            ]
            mock_db.commit = MagicMock()

            mock_session = MagicMock()
            mock_session.__enter__ = lambda self: mock_db
            mock_session.__exit__ = lambda self, *args: None

            with patch("apps.worker.tasks.SessionLocal", return_value=mock_session):
                try:
                    evaluate_dataset("ws-1", "run-1", "job-1", "rev-1", [1, 5, 10])
                except Exception:
                    pass

            # The provider passed to GroundedAnswerGenerator should be the factory's wrapped
            # provider (not wrapped again by the worker)
            mock_generator_class.assert_called_once()
            args, _ = mock_generator_class.call_args
            passed_provider = args[1]
            assert passed_provider is mock_provider
            # The provider should be the factory's output (already has retry/circuit/fallback)
            # Not wrapped again by worker code

    def test_existing_task_parameters_preserved(self):
        """Test that existing task parameters and top_k behavior remain unchanged."""
        from core.llm.base import GenerationSettings

        with (
            patch("apps.worker.tasks.get_chat_provider") as mock_get_provider,
            patch("eval.runners.evaluate.EvaluationRunner") as mock_runner,
            patch("apps.worker.tasks.HostedElementRetriever"),
            patch("apps.worker.tasks.GroundedAnswerGenerator") as mock_generator_class,
            patch("apps.worker.tasks.get_settings") as mock_get_settings,
            patch("apps.worker.tasks.SessionLocal") as mock_session_local,
        ):
            mock_provider, mock_revision = self._setup_common_mocks(
                mock_get_settings, mock_get_provider
            )

            mock_report = MagicMock()
            mock_report.status = "complete"
            mock_report.queries = [1, 2, 3]
            mock_report.model_dump_json.return_value = '{"status": "complete", "queries": [1,2,3]}'
            mock_runner.return_value.run.return_value = mock_report

            # Mock the answerer to return a proper GeneratedAnswer
            mock_answerer = MagicMock()
            mock_answerer.provider = mock_provider
            mock_answerer.settings = GenerationSettings(temperature=0.0, max_tokens=512)
            mock_generated_answer = MagicMock()
            mock_generated_answer.answer = "Test answer"
            mock_generated_answer.citations = []
            mock_generated_answer.unresolved_citation_labels = []
            mock_answerer.answer_with_sources.return_value = mock_generated_answer
            mock_answerer.answer.return_value = mock_generated_answer
            mock_generator_class.return_value = mock_answerer

            mock_run = MagicMock()
            mock_run.id = "run-1"
            mock_run.workspace_id = "ws-1"
            mock_run.status = "pending"

            mock_job = MagicMock()
            mock_job.id = "job-1"
            mock_job.workspace_id = "ws-1"

            mock_db = MagicMock()
            mock_db.bind.dialect.name = "sqlite"
            # Scalar calls: run, job, revision (first try), then run, job (after completion)
            mock_db.scalar.side_effect = [
                mock_run,
                mock_job,
                mock_revision,  # First try
                mock_run,
                mock_job,  # After successful completion
            ]
            # Also mock scalars for the elements query
            mock_scalars_result = MagicMock()
            mock_scalars_result.__iter__ = lambda self: iter([])
            mock_db.scalars.return_value = mock_scalars_result
            mock_db.commit = MagicMock()
            mock_db.rollback = MagicMock()

            mock_session = MagicMock()
            mock_session.__enter__ = lambda self: mock_db
            mock_session.__exit__ = lambda self, *args: None
            mock_session_local.return_value = mock_session

            evaluate_dataset("ws-1", "run-1", "job-1", "rev-1", [3, 7, 15])

            # Verify top_k is max_cutoff (15 from [3, 7, 15])
            mock_generator_class.assert_called_once()
            _, kwargs = mock_generator_class.call_args
            assert kwargs.get("top_k") == 15

            # Verify retriever is constructed with elements
            from apps.worker.tasks import HostedElementRetriever

            HostedElementRetriever.assert_called_once()

    def test_run_experiment_top_k_preserved(self):
        """Test that run_experiment preserves top_k from evaluation config."""
        from core.llm.base import GenerationSettings

        with (
            patch("apps.worker.tasks.get_chat_provider") as mock_get_provider,
            patch("eval.runners.experiment.ExperimentRunner") as mock_runner,
            patch("apps.worker.tasks.HostedElementRetriever"),
            patch("apps.worker.tasks.GroundedAnswerGenerator") as mock_generator_class,
            patch("apps.worker.tasks.get_settings") as mock_get_settings,
            patch("apps.worker.tasks.SessionLocal") as mock_session_local,
        ):
            mock_provider, mock_revision = self._setup_common_mocks(
                mock_get_settings, mock_get_provider
            )

            mock_report = MagicMock()
            mock_report.status = "complete"
            mock_report.variants = []
            mock_report.model_dump_json.return_value = '{"status": "complete", "variants": []}'
            mock_runner.return_value.run.return_value = mock_report

            # Mock the answerer to return a proper GeneratedAnswer
            mock_answerer = MagicMock()
            mock_answerer.provider = mock_provider
            mock_answerer.settings = GenerationSettings(temperature=0.0, max_tokens=512)
            mock_generated_answer = MagicMock()
            mock_generated_answer.answer = "Test answer"
            mock_generated_answer.citations = []
            mock_generated_answer.unresolved_citation_labels = []
            mock_answerer.answer_with_sources.return_value = mock_generated_answer
            mock_answerer.answer.return_value = mock_generated_answer
            mock_generator_class.return_value = mock_answerer

            mock_experiment = MagicMock()
            mock_experiment.id = "exp-1"
            mock_experiment.workspace_id = "ws-1"
            mock_experiment.status = "pending"

            mock_job = MagicMock()
            mock_job.id = "job-1"
            mock_job.workspace_id = "ws-1"

            mock_db = MagicMock()
            mock_db.bind.dialect.name = "sqlite"
            # Scalar calls: experiment, job, revision (first try), then
            # experiment, job (after completion)
            mock_db.scalar.side_effect = [
                mock_experiment,
                mock_job,
                mock_revision,  # First try
                mock_experiment,
                mock_job,  # After successful completion
            ]
            # Also mock scalars for the elements query
            mock_scalars_result = MagicMock()
            mock_scalars_result.__iter__ = lambda self: iter([])
            mock_db.scalars.return_value = mock_scalars_result
            mock_db.commit = MagicMock()
            mock_db.rollback = MagicMock()

            mock_session = MagicMock()
            mock_session.__enter__ = lambda self: mock_db
            mock_session.__exit__ = lambda self, *args: None
            mock_session_local.return_value = mock_session

            payload = {
                "cutoffs": [2, 8, 20],
                "baseline_variant": "baseline",
                "bootstrap_resamples": 100,
                "random_seed": 42,
                "variants": [
                    {"name": "baseline", "retrieval_depth": 5},
                ],
            }

            run_experiment("ws-1", "exp-1", "job-1", "rev-1", payload)

            # Verify top_k is max_cutoff (20 from [2, 8, 20])
            mock_generator_class.assert_called_once()
            _, kwargs = mock_generator_class.call_args
            assert kwargs.get("top_k") == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
