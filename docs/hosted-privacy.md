# RAG_EVAL Hosted Beta Privacy Notice

**Effective date: November 1, 2026**

This notice describes the invite-only hosted service operated by Ashmeet Singh. It supplements the repository's software privacy documentation and applies to hosted accounts.

## Information processed

- **GitHub account identity:** GitHub user ID, login, and verified primary email used to check invitations and provide sign-in. The OAuth app requests identity access, not repository access.
- **Workspace content:** uploaded UTF-8 text and Markdown files, extracted source elements, queries, dataset labels, generated answers, retrieved excerpts, evaluation reports, and experiment records.
- **Service data:** session records, job state, content digests, timestamps, and security/diagnostic logs.
- **Provider credentials:** optional Gemini or Groq chat API keys, encrypted before database storage. The service does not return the plaintext key after saving it.

## Use and disclosure

Workspace content is processed to index sources, retrieve evidence, answer questions, and run evaluations requested by the user. Server-managed Ollama handles embeddings and default chat. If a user selects Gemini or Groq for chat, the relevant prompt and retrieved source excerpts are sent to that provider. Those providers process data under their own terms. The service does not sell workspace content.

## Retention and deletion

Original files are retained privately until the user deletes a document. Document deletion removes the original and active index; previous reports may keep excerpts for reproducibility. Workspace deletion is intended to remove active documents, indexes, credentials, datasets, jobs, and reports. Encrypted backups may retain data for up to 30 days before expiry. Operational logs should avoid content and credentials and follow a separately configured retention schedule.

## Security

The service uses private object storage, workspace-scoped application queries, secure session cookies when deployed behind HTTPS, and encryption for saved provider credentials. No internet system can guarantee absolute security. Production launch requires database row-level security, rate limiting, CSRF protection, tested deletion/backup expiry, and operational access controls.

## Your choices and contact

Users can remove saved provider credentials and delete documents in the application. For account or workspace deletion requests, contact the project owner through the RAG_EVAL repository. The beta currently has no self-service data export or workspace deletion interface.
