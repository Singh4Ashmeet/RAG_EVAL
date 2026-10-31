# Contributing to RAG_EVAL

RAG_EVAL is a proprietary project maintained by Ashmeet Singh. Please contact the maintainer before spending time on a substantial change or submitting code. By contributing through the public GitHub repository, you also agree to GitHub's [Terms of Service](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service), including its terms for content added to a repository with a license notice. Any separate transfer, broader license, or different contribution terms must be agreed in writing before a contribution is accepted. Do not submit code unless you have the right to license it on those terms.

## Before proposing a change

1. Open or discuss an issue describing the problem and proposed scope.
2. Confirm that the change fits the documented [project scope](docs/architecture.md#current-scope-and-limitations).
3. Do not include credentials, private documents, model files, generated evaluation reports, or unreviewed ground truth.

## Development setup

```powershell
uv sync
uv run task test
uv run task typecheck
uv run task lint
uv run ruff format --check .
```

The test suite is designed to run without Ollama or model downloads. For local provider changes, add deterministic fake-client tests. Do not make tests depend on paid APIs, network access, or installed model weights.

## Change expectations

- Preserve existing public contracts unless the change documents a migration.
- Add tests for behavior and failure cases.
- Keep evaluation data synthetic in tests. Project benchmark labels must be reviewed before addition to `eval/golden/questions.jsonl`.
- Update the relevant architecture, runbook, benchmark, or findings documentation.
- Report the commands run and observed outcomes in the pull request.

## Pull requests

Keep changes focused and explain the user-visible behavior, implementation, tests, and known limitations. Do not claim measured improvements without reviewed ground truth and a controlled experiment. The maintainer must agree to the applicable contribution license in writing before merging.
