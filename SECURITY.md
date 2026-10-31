# Security policy

## Supported project state

Security fixes are handled on the current `main` branch. No release or backport support policy is currently published.

## Reporting a vulnerability

Do not publish credentials, private documents, exploit details, or an unpatched vulnerability in a public issue. If GitHub private vulnerability reporting is enabled for this repository, use its private reporting flow. Otherwise, contact the maintainer through the [RAG_EVAL GitHub profile](https://github.com/Singh4Ashmeet) and request a private channel before sending technical details.

Please include the affected revision, impact, reproduction steps, and any mitigation you have identified. The maintainer will acknowledge reports when practicable and coordinate disclosure after a fix or mitigation is available.

## Handling sensitive data

Do not commit `.env` files, API keys, local databases, uploaded documents, or generated reports containing sensitive data. Review the configured Ollama host and any connected service before sending private content; a non-local host may receive prompts and retrieved passages. See [the privacy and data handling notice](docs/privacy.md).
