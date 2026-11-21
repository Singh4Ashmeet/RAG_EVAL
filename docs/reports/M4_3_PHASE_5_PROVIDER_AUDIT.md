# M4.3 Phase 5 Provider Audit Report

## Current Architecture Overview

### LLM Provider Interface (`core/llm/base.py`)

The system defines a `LLMProvider` protocol with:
- `model_id: str` property - stable provider/model identifier
- `generate(system_prompt, user_prompt, settings: GenerationSettings) -> str` method

`GenerationSettings`:
- `temperature`: float (0.0-2.0), default 0.0
- `max_tokens`: int (>=1), default 512

### Current Providers

| Provider | File | Status | Implementation |
|----------|------|--------|----------------|
| Ollama | `ollama_provider.py` | ✅ Complete | Uses `ollama` Python SDK |
| Gemini | `gemini_provider.py` | ❌ Placeholder only | Empty file |
| Groq | `groq_provider.py` | ❌ Placeholder only | Empty file |

### Provider Construction & Usage

**OllamaLLMProvider** (`core/llm/ollama_provider.py`):
- Constructor: `model: str, host: str | None = None, client: Any | None = None, timeout: float | None = 180.0`
- `model_id` = `f"ollama:{cleaned_model}"`
- Uses `ollama.Client` SDK
- `generate()` calls `client.chat()` with system/user messages
- Returns `content.strip()` from response

**API Integration** (`apps/api/main.py` - `/ask` endpoint):
- Uses `payload.provider` ("ollama", "gemini", "groq")
- For Ollama: direct HTTP to `/api/chat`
- For Gemini/Groq: Uses encrypted credentials from DB, direct HTTP to provider APIs
- Provider selection via dict mapping: `{"ollama": settings.chat_model, "gemini": settings.gemini_chat_model, "groq": settings.groq_chat_model}`

**Worker Tasks** (`apps/worker/tasks.py`):
- `HostedElementRetriever` uses Ollama embeddings via direct HTTP
- `evaluate_dataset()` and `run_experiment()` create `GroundedAnswerGenerator` with `OllamaLLMProvider`
- Provider instantiated with `model=settings.chat_model`, `host=settings.chat_base_url`

**Configuration** (`core/config.py`):
```python
chat_provider: Literal["ollama", "gemini", "groq"] = "ollama"
chat_model: str = "llama3.2:3b"
chat_base_url: str = "http://localhost:11434"
chat_api_key: str = ""
gemini_chat_model: str = "gemini-3.8-flash"
groq_chat_model: str = "openai/gpt-oss-120b"
```

### API Response Format (`AskResponse`)

```python
class AskResponse(BaseModel):
    answer: str
    sources: list[AskSource]
    citations: list[dict[str, str]]
    unresolved_citation_labels: list[str]
    provider: str  # provider.title()
    model: str
    citation_labels: list[str]  # ["S1", "S2", ...]
```

### Embedding Provider

Only Ollama is supported (`EMBEDDING_PROVIDER = "ollama"`). `OllamaEmbedder` uses direct HTTP to `/api/embed`.

### Worker Inference

Worker tasks create `GroundedAnswerGenerator` with `OllamaLLMProvider`:
- `evaluate_dataset()` and `run_experiment()` use `OllamaLLMProvider(model=settings.chat_model, host=settings.chat_base_url)`
- `HostedElementRetriever` uses `OllamaEmbedder` via HTTP to `/api/embed`

### Dependencies (pyproject.toml)
- `google-genai>=2.34.0` - Google Generative AI SDK
- `groq>=1.7.0` - Groq Python SDK
- `ollama>=0.6.3` - Ollama Python SDK
- `httpx>=0.28.1` - HTTP client

### Test Coverage
- `test_ollama_provider.py`: 8 tests covering provider behavior
- No tests for Gemini/Groq providers (placeholders)

---

## Gaps Identified

1. **GeminiProvider** - Empty placeholder, needs full implementation
2. **GroqProvider** - Empty placeholder, needs full implementation  
3. **Provider Registry** (`core/llm/router.py`) - Empty placeholder, needs implementation
4. **Provider Registry Integration** - API and worker don't use a registry
4. **Rate Limiting / Retry Logic** - Not implemented
5. **Retry-After Handling** - Not implemented
6. **Error Type Normalization** - No standardized exception hierarchy
7. **Tests for new providers** - None exist
8. **Worker provider selection** - Still hardcoded to Ollama
8. **Gemini embeddings** - Not implemented (config says "currently only ollama")
9. **Groq embeddings** - Not supported (Groq doesn't offer embeddings API)

---

## Implementation Plan

### Phase 1: Provider Registry (`core/llm/router.py`)
- Create `get_chat_provider()` and `get_embedding_provider()` functions
- Read from centralized Settings
- Return appropriate provider instances

### Phase 2: Implement GeminiProvider
- Use `google-genai` SDK
- Match `LLMProvider` protocol
- Handle auth, rate limits, timeouts

### Phase 3: Implement GroqProvider  
- Use `groq` SDK
- Match `LLMProvider` protocol
- Chat-only (no embeddings)

### Phase 4: Error Handling & Retry Logic
- Define exception hierarchy
- Implement retry with exponential backoff
- Handle `Retry-After` headers

### Phase 5: Integration
- Update `/ask` endpoint to use registry
- Update worker tasks to use registry
- Add tests

### Phase 6: Tests & Documentation
- Unit tests for providers
- Integration tests (mocked)
- Update .env.example
- Documentation