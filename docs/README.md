# RAG_EVAL documentation

Start with the root [README](../README.md) for project scope, installation, and the shortest end-to-end example.

## Guides

- [Architecture](architecture.md): modules, data flow, persistence rules, and implementation limits.
- [Runbook](runbook.md): configure Ollama, ingest/index documents, generate answers, evaluate, compare variants, and diagnose runs.
- [Benchmarks and metrics](benchmarks.md): metric definitions, experiment interpretation, and historical model-inference measurements.
- [Findings and limitations](findings.md): verified capabilities, current gaps, and evidence still needed.
- [Privacy and data handling](privacy.md): documents, prompts, vectors, reports, and configured model hosts.
- [Development runner and CI/CD](ci-cd.md): start both local services and understand the GitHub Actions pipeline.

## Project policies

- [License](../LICENSE): proprietary notice and copyright attribution.
- [Terms and Conditions](../TERMS.md): intended use and warranty terms.
- [Contributing](../CONTRIBUTING.md): development and review expectations.
- [Security](../SECURITY.md): responsible vulnerability reporting.
- [Changelog](../CHANGELOG.md): milestone-level project history.

## Suggested reading paths

- **First local run:** root README → runbook prerequisites → dense retrieval and answer examples.
- **Build an evaluation dataset:** runbook JSONL section → benchmark metric definitions.
- **Compare configurations:** runbook experiment section → benchmark interpretation notes.
- **Plan a user interface:** architecture → findings → root README current scope.

## Ground-truth policy

`eval/golden/questions.jsonl` is intentionally empty. Add project questions and expected chunk IDs/reference answers only after they have been reviewed. Synthetic test data verifies implementation behavior, but it is not a project benchmark and must not be reported as one.
