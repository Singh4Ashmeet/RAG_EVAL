# Privacy and data handling

This page describes the local developer workflow. The separate invite-only hosted beta has account, tenant, retention, and provider-processing behavior described in the [Hosted Privacy Notice](hosted-privacy.md) and [Hosted Beta Guide](hosted-beta.md).

## Data handled by the current workflow

- The ingestion path reads `.txt` and `.md` files from paths supplied by the operator and extracts their text and metadata.
- The SQLite vector store persists chunks and embedding vectors in the configured database file.
- Retrieval sends a query to the configured embedding provider and reads matching stored chunks.
- Answer generation sends the query and retrieved passages to the configured chat provider.
- Evaluation and experiment reports can include query text, expected answers, retrieved passages, generated answers, citations, errors, timing, and model identifiers.

The default Ollama provider is intended for a local Ollama service. If `OLLAMA_HOST` points elsewhere, the configured host receives the embedding inputs or chat prompt. RAG_EVAL cannot determine how that host retains or processes requests.

## Telemetry and dependencies

The local workflow contains no project analytics or telemetry component. Package installation, model runtimes, operating systems, and services may perform their own network communication; review their documentation and configuration. The hosted beta processes uploaded documents and prompts on the service and can send chat prompts with retrieved excerpts to Gemini or Groq when a tenant selects that provider.

## Operator responsibilities

- Process only data you are authorized to use with the selected models and services.
- Protect the SQLite database, source documents, environment variables, and reports.
- Avoid committing sensitive evaluation data. `eval/reports/` is ignored by Git by default, but inspect repository status before publishing.
- Use synthetic fixtures in tests and keep reviewed project ground truth separate from private data.
- Delete local data and model files using the relevant tools for your environment; removing a document from a source directory does not automatically remove its indexed chunks.

This notice describes observed project behavior; it is not a guarantee about third-party components or a substitute for a deployment-specific privacy assessment.
