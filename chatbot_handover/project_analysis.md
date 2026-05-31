# Phase 1 — Project Analysis: Chatbot Module

> **Document Type**: Project Analysis  
> **Generated**: 2026-05-30  
> **Scope**: Complete chatbot implementation within the Drug Response Prediction and Recommendation Platform

---

## 1. Executive Summary

The chatbot module is an AI-powered conversational assistant integrated into the Drug Response Prediction and Recommendation Platform — a graduation project in bioinformatics. It leverages Google's Gemini generative AI models to provide domain-aware explanations about IC50 values, drug sensitivity, recommendation scores, preprocessing pipelines, and ML model architecture. The system features a robust fallback mechanism, conversation memory management, model fallback on quota errors (429), and strict safety restrictions preventing medical diagnosis or treatment recommendations.

---

## 2. File Inventory

### 2.1 Backend Core (chatbot_module/backend/)

| File | Path | Purpose |
|------|------|---------|
| `chatbot_service.py` | [`chatbot_module/backend/chatbot_service.py`](chatbot_module/backend/chatbot_service.py) | Core service: Gemini integration, fallback logic, model fallback chain, singleton pattern |
| `chatbot_router.py` | [`chatbot_module/backend/chatbot_router.py`](chatbot_module/backend/chatbot_router.py) | FastAPI router: 4 endpoints (POST /chat, GET /history/{id}, DELETE /history/{id}, GET /status) |
| `chatbot_schema.py` | [`chatbot_module/backend/chatbot_schema.py`](chatbot_module/backend/chatbot_schema.py) | Pydantic schemas: ChatRequest, ChatResponse, ChatMessage, ChatHistoryResponse, ChatbotStatusResponse, ErrorResponse |
| `chatbot_memory.py` | [`chatbot_module/backend/chatbot_memory.py`](chatbot_module/backend/chatbot_memory.py) | Conversation memory: in-memory storage, max 50 messages/conversation, Gemini history format conversion |
| `chatbot_utils.py` | [`chatbot_module/backend/chatbot_utils.py`](chatbot_module/backend/chatbot_utils.py) | Utilities: .env loading, fallback response dictionary, topic keyword matching |
| `__init__.py` | [`chatbot_module/backend/__init__.py`](chatbot_module/backend/__init__.py) | Package init |

### 2.2 API Integration Layer (api/)

| File | Path | Purpose |
|------|------|---------|
| `chatbot.py` (router) | [`api/routers/chatbot.py`](api/routers/chatbot.py) | Production router mounted in main FastAPI app — mirrors chatbot_module router |
| `chatbot.py` (schema) | [`api/schemas/chatbot.py`](api/schemas/chatbot.py) | Production schemas — mirrors chatbot_module schemas (minus ErrorResponse) |
| `chatbot_service.py` | [`api/services/chatbot_service.py`](api/services/chatbot_service.py) | Production service — standalone version with inline memory management (not using ConversationMemory class) |

### 2.3 Prompts (chatbot_module/prompts/)

| File | Path | Purpose |
|------|------|---------|
| `system_prompt.txt` | [`chatbot_module/prompts/system_prompt.txt`](chatbot_module/prompts/system_prompt.txt) | Full domain-specific system prompt (86 lines) loaded at runtime |
| `chatbot_behavior.txt` | [`chatbot_module/prompts/chatbot_behavior.txt`](chatbot_module/prompts/chatbot_behavior.txt) | Communication style, response format, conversation continuity, context handling, fallback behavior rules |
| `safety_rules.txt` | [`chatbot_module/prompts/safety_rules.txt`](chatbot_module/prompts/safety_rules.txt) | Mandatory safety restrictions with enforcement details and example disclaimers |

### 2.4 Frontend Integration (chatbot_module/frontend_integration/)

| File | Path | Purpose |
|------|------|---------|
| `REACT_INTEGRATION.md` | [`chatbot_module/frontend_integration/REACT_INTEGRATION.md`](chatbot_module/frontend_integration/REACT_INTEGRATION.md) | React component code, ChatSession class, state management patterns, CSS |
| `API_EXAMPLES.md` | [`chatbot_module/frontend_integration/API_EXAMPLES.md`](chatbot_module/frontend_integration/API_EXAMPLES.md) | Fetch/axios/React examples, TypeScript type definitions, ChatSession utility class |
| `CHAT_FLOW.md` | [`chatbot_module/frontend_integration/CHAT_FLOW.md`](chatbot_module/frontend_integration/CHAT_FLOW.md) | Message sequence diagrams (new conversation, continued, with context, error recovery, fallback) |
| `SAMPLE_REQUESTS.json` | [`chatbot_module/frontend_integration/SAMPLE_REQUESTS.json`](chatbot_module/frontend_integration/SAMPLE_REQUESTS.json) | Complete sample request/response payloads for all endpoints and validation errors |

### 2.5 Tests (chatbot_module/tests/)

| File | Path | Purpose |
|------|------|---------|
| `test_chatbot.py` | [`chatbot_module/tests/test_chatbot.py`](chatbot_module/tests/test_chatbot.py) | 20+ tests: ConversationMemory, fallback responses, ChatbotService, schema validation |
| `test_gemini.py` | [`chatbot_module/tests/test_gemini.py`](chatbot_module/tests/test_gemini.py) | 15+ tests: Gemini init, API calls, role mapping, empty response, model fallback chain, context enhancement |
| `test_api.py` | [`chatbot_module/tests/test_api.py`](chatbot_module/tests/test_api.py) | 12+ tests: All 4 endpoints, request validation, error handling, history management |

### 2.6 Configuration

| File | Path | Purpose |
|------|------|---------|
| `.env` | [`.env`](.env) | Active config: GEMINI_API_KEY, GEMINI_MODEL, API_HOST/PORT, DEFAULT_MODEL, CORS |
| `.env.example` | [`chatbot_module/.env.example`](chatbot_module/.env.example) | Template with documentation for all environment variables |
| `__init__.py` | [`chatbot_module/__init__.py`](chatbot_module/__init__.py) | Package metadata: version="1.0.0" |

---

## 3. Architecture Overview

### 3.1 Dual Implementation Pattern

The project contains **two parallel implementations** of the chatbot service:

1. **Modular Implementation** (`chatbot_module/backend/`) — Refactored, clean separation of concerns:
   - `ChatbotService` uses `ConversationMemory` class for history
   - `chatbot_utils` provides `FALLBACK_RESPONSES` dict and `match_fallback_topic()` function
   - `chatbot_schema` provides all Pydantic models including `ErrorResponse`
   - System prompt loaded from `prompts/system_prompt.txt` at runtime with embedded fallback

2. **Inline Implementation** (`api/services/chatbot_service.py`) — Original monolithic version:
   - `ChatbotService` manages conversations inline (self.conversations dict)
   - `FALLBACK_RESPONSES` and `_match_fallback_topic()` defined in the same file
   - System prompt is an inline `SYSTEM_PROMPT` constant (no file loading)
   - No `ConversationMemory` class — history management is embedded in the service

Both implementations share the **same core logic**: Gemini API integration, model fallback chain, context enhancement, fallback responses, and singleton pattern.

### 3.2 Router Duplication

Similarly, there are two router implementations:
- [`chatbot_module/backend/chatbot_router.py`](chatbot_module/backend/chatbot_router.py) — uses modular imports
- [`api/routers/chatbot.py`](api/routers/chatbot.py) — uses api-layer imports

The main app ([`api/main.py`](api/main.py:126)) includes the api-layer router:
```python
app.include_router(chatbot.router)
```

---

## 4. Component Analysis

### 4.1 ChatbotService (Modular — chatbot_module/backend/chatbot_service.py)

**Class**: `ChatbotService` (lines 159–461)

**Attributes**:
| Attribute | Type | Default | Purpose |
|-----------|------|---------|---------|
| `gemini_model` | `GenerativeModel or None` | `None` | Gemini API model instance |
| `api_key_configured` | `bool` | `False` | Whether GEMINI_API_KEY is valid |
| `gemini_model_name` | `str` | `os.getenv("GEMINI_MODEL", "gemini-2.5-flash")` | Active Gemini model name |
| `_fallback_models` | `list` | `["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash"]` | Models to try on 429 errors |
| `memory` | `ConversationMemory` | `ConversationMemory()` | Conversation history manager |
| `_system_prompt` | `str` | Loaded from file or embedded | Domain-specific system prompt |

**Methods**:
| Method | Line | Purpose |
|--------|------|---------|
| `__init__()` | 178 | Initialize service, load prompt, call `_initialize_gemini()` |
| `_initialize_gemini()` | 192 | Configure Gemini API with key from environment |
| `is_available()` | 237 | Check if Gemini API is configured and model exists |
| `get_status()` | 246 | Return status dict (available, model, api_key_configured, message) |
| `chat()` | 264 | Main method: process message, try Gemini with fallback chain, or use fallback responses |
| `_call_gemini()` | 345 | Call Gemini API with conversation history |
| `_call_gemini_with_model()` | 393 | Call Gemini API with specific model name (for fallback) |
| `get_conversation_history()` | 439 | Delegate to `memory.get_formatted_history()` |
| `clear_conversation()` | 451 | Delegate to `memory.clear_conversation()` |

**Singleton**: `get_chatbot_service()` (line 471) — returns global `_chatbot_service_instance`

### 4.2 ConversationMemory (chatbot_module/backend/chatbot_memory.py)

**Class**: `ConversationMemory` (lines 22–155)

**Attributes**:
| Attribute | Type | Default | Purpose |
|-----------|------|---------|---------|
| `conversations` | `Dict[str, List[Dict]]` | `{}` | In-memory conversation storage keyed by conversation_id |
| `max_history` | `int` | `50` | Maximum messages per conversation |

**Methods**:
| Method | Line | Purpose |
|--------|------|---------|
| `add_message()` | 44 | Add message to history, trim if over max_history |
| `get_history()` | 68 | Return raw message list for a conversation_id |
| `get_formatted_history()` | 80 | Return API-formatted dict with role, content, timestamp |
| `build_gemini_history()` | 105 | Convert to Gemini API format: role "assistant" → "model", add "parts" structure |
| `clear_conversation()` | 127 | Delete conversation from memory |
| `get_active_conversation_count()` | 148 | Return number of stored conversations |

### 4.3 Chatbot Utils (chatbot_module/backend/chatbot_utils.py)

**Components**:
| Component | Line | Purpose |
|-----------|------|---------|
| `.env` loading | 22–32 | Load dotenv with explicit path, override=True |
| `FALLBACK_RESPONSES` | 39–79 | Dict with 5 topic keys: ic50, recommendation, preprocessing, models, general |
| `match_fallback_topic()` | 82–103 | Keyword matching function returning topic key |

**Fallback Topic Keywords**:
| Topic | Keywords |
|-------|----------|
| `ic50` | ic50, ic 50, half maximal, inhibitory concentration, sensitivity, ln_ic50 |
| `recommendation` | recommend, ranking, score, drug order, best drug, top drug |
| `preprocessing` | preprocess, encoding, label encod, scaling, feature selection, pipeline |
| `models` | model, catboost, xgboost, lightgbm, gradient, ridge, lasso, elasticnet |
| `general` | (default — anything not matching above) |

### 4.4 Chatbot Schemas (chatbot_module/backend/chatbot_schema.py)

**Models**:
| Model | Line | Fields | Validation |
|-------|------|--------|------------|
| `ChatRequest` | 20 | message (str, 1–2000), conversation_id (Optional[str]), context (Optional[Dict]) | message stripped, non-empty |
| `ChatMessage` | 63 | role (str), content (str), timestamp (datetime) | role must be "user" or "assistant" |
| `ChatResponse` | 99 | success (bool), message (str), conversation_id (Optional[str]), timestamp (datetime), model_used (Optional[str]) | — |
| `ChatHistoryResponse` | 120 | success (bool), conversation_id (Optional[str]), messages (List[ChatMessage]), timestamp (datetime) | — |
| `ChatbotStatusResponse` | 150 | available (bool), model (Optional[str]), api_key_configured (bool), message (str) | — |
| `ErrorResponse` | 169 | success (bool=False), error (str), detail (Optional[str]) | — |

### 4.5 Chatbot Router (chatbot_module/backend/chatbot_router.py)

**Endpoints**:
| Endpoint | Method | Line | Response Model |
|----------|--------|------|----------------|
| `POST /chat` | `chat_with_assistant()` | 44 | `ChatResponse` |
| `GET /chat/history/{conversation_id}` | `get_conversation_history()` | 92 | `ChatHistoryResponse` |
| `DELETE /chat/history/{conversation_id}` | `clear_conversation_history()` | 121 | JSON dict |
| `GET /chat/status` | `chatbot_status()` | 146 | `ChatbotStatusResponse` |

**Router Config**: prefix="/chat", tags=["chatbot"], error responses: 400, 500

---

## 5. Gemini Integration Analysis

### 5.1 API Key Loading

The Gemini API key is loaded from environment variables via three mechanisms:
1. **Main app .env loading** ([`api/main.py`](api/main.py:27–36)): `load_dotenv()` with explicit path before router imports
2. **Chatbot utils .env loading** ([`chatbot_module/backend/chatbot_utils.py`](chatbot_module/backend/chatbot_utils.py:22–32)): `load_dotenv()` with override=True
3. **Chatbot service .env loading** ([`api/services/chatbot_service.py`](api/services/chatbot_service.py:24–34)): `load_dotenv()` with override=True

The key is validated in `_initialize_gemini()`:
- Empty string → fallback mode
- Placeholder `"your_gemini_api_key_here"` → fallback mode
- Valid key → `genai.configure(api_key=api_key)` + model creation

### 5.2 Model Selection

**Primary model**: `os.getenv("GEMINI_MODEL", "gemini-2.5-flash")`

**Fallback chain**: `["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash"]`

The fallback chain works as follows:
1. Build `models_to_try` list: primary model + fallback models (deduplicated)
2. For each model, call `_call_gemini_with_model()`
3. If 429/quota error → try next model
4. If non-quota error → break, fall to fallback responses
5. If success → update `gemini_model_name` to the working model, return response

### 5.3 System Prompt Engineering

The system prompt (86 lines) is structured with:
- **Capabilities section**: Lists what the chatbot can explain
- **Project Architecture Knowledge**: Preprocessing pipeline (3 stages), prediction pipeline, recommendation engine, IC50 interpretation thresholds, available models, data sources
- **Safety Restrictions**: 6 mandatory rules preventing medical diagnosis/treatment
- **Communication Style**: Scientific but accessible, use analogies, specific numbers, formatting
- **Response Format**: Structured format for prediction explanations and concept explanations

The prompt is loaded from [`chatbot_module/prompts/system_prompt.txt`](chatbot_module/prompts/system_prompt.txt) at runtime. If the file is missing, an embedded `SYSTEM_PROMPT` constant (identical content) is used as fallback.

### 5.4 Conversation Memory for Gemini

The `ConversationMemory.build_gemini_history()` method converts internal format to Gemini API format:
- Role mapping: `"assistant"` → `"model"` (Gemini requires "model" not "assistant")
- Structure: `{"role": "user"|"model", "parts": [content]}` (Gemini requires "parts" list)

History is passed to `model.start_chat(history=chat_history)` to maintain conversation context.

### 5.5 Context Enhancement

When `context` is provided in the chat request, the service builds an enhanced message:
```
[Context: predicted_ic50: -1.46, model_used: CatBoost, cell_line_name: A172, drug_name: Camptothecin]

User question: What does this prediction mean?
```

This allows Gemini to provide specific, tailored responses based on actual prediction data.

---

## 6. Fallback System Analysis

### 6.1 Three-Layer Fallback Architecture

| Layer | Trigger | Response |
|-------|---------|----------|
| **Layer 1**: Model fallback | 429 quota error on primary model | Try next model in fallback chain |
| **Layer 2**: Fallback responses | Gemini completely unavailable (no API key, import error, all models quota-exhausted, non-quota error) | Return pre-defined topic-matched response |
| **Layer 3**: Empty response handling | Gemini returns empty response (safety filter) | Return polite apology message about safety filters |

### 6.2 Fallback Response Characteristics

- Each fallback response includes a **⚠️ Note** informing the user that Gemini AI is unavailable
- Responses are **domain-accurate** — they contain correct information about IC50, preprocessing, etc.
- The `general` fallback response advises configuring `GEMINI_API_KEY`
- All fallback responses include the **educational/research purposes only** disclaimer

### 6.3 Model Fallback Switching

When a fallback model succeeds:
- `self.gemini_model_name` is **permanently updated** to the working model
- This means subsequent requests will use the fallback model as primary
- Log message: `[CHAT] Switched active model from {old} to {new}`

---

## 7. Session Management Analysis

### 7.1 Conversation ID Generation

- If `conversation_id` is provided → use it (continues existing conversation)
- If not provided → generate `str(uuid.uuid4())` (new conversation)
- The ID is returned in every response for the frontend to store

### 7.2 History Storage

- **Storage type**: In-memory Python dictionary (`self.conversations` or `ConversationMemory.conversations`)
- **Key**: conversation_id (string)
- **Value**: List of message dicts with role, content, timestamp
- **Max size**: 50 messages per conversation (older messages trimmed)
- **Persistence**: None — conversations are lost on server restart

### 7.3 History API

| Operation | Method | Endpoint |
|-----------|--------|----------|
| Retrieve | `GET /chat/history/{conversation_id}` | Returns formatted message list |
| Clear | `DELETE /chat/history/{conversation_id}` | Deletes conversation from memory |

---

## 8. Safety System Analysis

### 8.1 Prompt-Level Safety

The system prompt includes 6 mandatory safety restrictions:
1. MUST NOT provide medical diagnosis
2. MUST NOT prescribe treatments or drug recommendations for real patients
3. MUST NOT suggest predictions should replace clinical decisions
4. MUST always clarify this is an educational/research tool
5. MUST encourage users to consult healthcare professionals
6. MUST redirect to qualified oncologist if asked about treating real patients

### 8.2 Gemini Safety Filters

Gemini API has built-in safety filters that may block responses:
- If `response.text` is None/empty → the service checks `response.candidates[0].finish_reason`
- Returns a polite apology: "I couldn't generate a response for that query. This may be due to content safety filters."

### 8.3 Fallback Response Safety

All fallback responses include:
- "⚠️ This tool is for educational and research purposes only. It does not provide medical advice."

---

## 9. Environment Configuration

### 9.1 Required Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `GEMINI_API_KEY` | Yes (for AI mode) | `""` | Google Gemini API authentication key |
| `GEMINI_MODEL` | No | `"gemini-2.5-flash"` | Primary Gemini model name |

### 9.2 Optional Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_HOST` | `"0.0.0.0"` | Server host |
| `API_PORT` | `"8000"` | Server port |
| `API_RELOAD` | `"True"` | Auto-reload on code changes |
| `API_LOG_LEVEL` | `"info"` | Logging level |
| `DEFAULT_MODEL` | `"CatBoost"` | Default ML prediction model |
| `CORS_ORIGINS` | `"*"` | CORS allowed origins |

### 9.3 Current Configuration (from .env)

- `GEMINI_API_KEY=AIzaSyB890fhMjtK38RIlMIgYKOJPr2q4AoNTLA` (active key)
- `GEMINI_MODEL=gemini-2.5-flash`

---

## 10. API Endpoints Summary

| Endpoint | Method | Request Body | Response | Status Codes |
|----------|--------|--------------|----------|--------------|
| `/chat` | POST | `ChatRequest` | `ChatResponse` | 200, 422, 500 |
| `/chat/status` | GET | — | `ChatbotStatusResponse` | 200, 500 |
| `/chat/history/{id}` | GET | — | `ChatHistoryResponse` | 200, 500 |
| `/chat/history/{id}` | DELETE | — | `{success, message}` | 200, 500 |

---

## 11. Test Coverage Summary

| Test Class | File | Tests | Coverage |
|------------|------|-------|----------|
| `TestConversationMemory` | [`test_chatbot.py`](chatbot_module/tests/test_chatbot.py) | 8 | add_message, max_history, formatted_history, gemini_history, clear, nonexistent, count, empty |
| `TestFallbackResponses` | [`test_chatbot.py`](chatbot_module/tests/test_chatbot.py) | 6 | ic50/recommendation/preprocessing/models/general topic matching, response content |
| `TestChatbotService` | [`test_chatbot.py`](chatbot_module/tests/test_chatbot.py) | 11 | basic chat, context, conversation_id, new ID, fallback mode, quota fallback, status, history, clear, singleton |
| `TestSchemaValidation` | [`test_chatbot.py`](chatbot_module/tests/test_chatbot.py) | 6 | valid request, empty message, too long, strip whitespace, role validation, response schema |
| `TestGeminiInitialization` | [`test_gemini.py`](chatbot_module/tests/test_gemini.py) | 6 | with/without key, placeholder key, import error, system instruction, model name config |
| `TestGeminiAPICalls` | [`test_gemini.py`](chatbot_module/tests/test_gemini.py) | 6 | basic call, with history, role mapping, empty response, model fallback, quota error |
| `TestModelFallbackChain` | [`test_gemini.py`](chatbot_module/tests/test_gemini.py) | 4 | chain on quota, all exhausted, non-quota error, model switching |
| `TestContextEnhancement` | [`test_gemini.py`](chatbot_module/tests/test_gemini.py) | 2 | with context, without context |
| `TestChatEndpoint` | [`test_api.py`](chatbot_module/tests/test_api.py) | 6 | basic, with context, with conversation_id, empty validation, too long, whitespace |
| `TestChatStatusEndpoint` | [`test_api.py`](chatbot_module/tests/test_api.py) | 2 | available, fallback mode |
| `TestChatHistoryEndpoint` | [`test_api.py`](chatbot_module/tests/test_api.py) | 2 | existing, unknown conversation |
| `TestChatHistoryClearEndpoint` | [`test_api.py`](chatbot_module/tests/test_api.py) | 2 | existing, nonexistent |

**Total**: ~57 tests across 3 files

---

## 12. Dependency Map

```
chatbot_module/backend/chatbot_service.py
  ├── imports from: chatbot_module.backend.chatbot_utils (FALLBACK_RESPONSES, match_fallback_topic)
  ├── imports from: chatbot_module.backend.chatbot_memory (ConversationMemory)
  ├── imports from: chatbot_module.prompts.system_prompt.txt (via _load_system_prompt())
  ├── imports from: google.generativeai (conditional, at runtime)
  ├── imports from: os, logging, uuid, pathlib
  └
chatbot_module/backend/chatbot_router.py
  ├── imports from: chatbot_module.backend.chatbot_schema (all schemas)
  ├── imports from: chatbot_module.backend.chatbot_service (get_chatbot_service)
  ├── imports from: fastapi (APIRouter, HTTPException, status)
  └
chatbot_module/backend/chatbot_memory.py
  ├── imports from: logging, datetime
  └
chatbot_module/backend/chatbot_utils.py
  ├── imports from: dotenv (load_dotenv), pathlib
  ├── imports from: os, logging
  └
chatbot_module/backend/chatbot_schema.py
  ├── imports from: pydantic (BaseModel, Field, field_validator)
  ├── imports from: datetime, typing
  └
api/main.py
  ├── imports from: api.routers.chatbot (router)
  ├── includes router: chatbot.router → mounted at /chat prefix
```

---

## 13. Key Design Patterns

| Pattern | Implementation | Purpose |
|---------|----------------|---------|
| **Singleton** | `get_chatbot_service()` with global `_chatbot_service_instance` | Ensure single service instance across the app |
| **Strategy (Fallback)** | Model fallback chain + fallback responses | Graceful degradation when Gemini unavailable |
| **Template Method** | System prompt loaded from file with embedded fallback | Configurable prompt with resilience |
| **Adapter** | `build_gemini_history()` converts internal format to Gemini format | Bridge between internal storage and external API |
| **Decorator (Context Enhancement)** | Context data prepended to user message | Enrich Gemini input without changing API contract |

---

## 14. Known Issues & Observations

1. **Dual Implementation**: Two parallel service implementations exist (modular vs inline). The main app uses the inline version from `api/services/chatbot_service.py`, not the modular one from `chatbot_module/backend/`.

2. **In-Memory Storage**: Conversation history is stored in Python dicts — lost on server restart. No database persistence.

3. **Model Switch Persistence**: When fallback model succeeds, `gemini_model_name` is permanently updated. This means the primary model won't be retried until server restart.

4. **No Rate Limiting**: No rate limiting on chat endpoints — vulnerable to abuse.

5. **No Authentication**: Chat endpoints have no authentication — anyone can send messages.

6. **CORS Wide Open**: `allow_origins=["*"]` — suitable for development but not production.

7. **API Key in .env**: The Gemini API key is stored in plaintext in `.env` — not encrypted.

8. **Frontend Not Integrated**: The current `web/js/app.js` does not include chatbot functionality — only prediction and recommendation features.

---

## 15. Technology Stack

| Component | Technology | Version/Details |
|-----------|------------|-----------------|
| Web Framework | FastAPI | Python async web framework |
| AI Model | Google Gemini | gemini-2.5-flash (primary), gemini-2.0-flash-lite, gemini-2.0-flash (fallbacks) |
| AI SDK | google-generativeai | Python SDK for Gemini API |
| Validation | Pydantic v2 | BaseModel, Field, field_validator |
| Environment | python-dotenv | .env file loading |
| Server | Uvicorn | ASGI server |
| Testing | pytest + FastAPI TestClient | Unit and integration tests |
| Frontend | Vanilla JS / React (guides) | Web interface + integration docs |

---

*This analysis document serves as the foundation for all subsequent documentation phases.*