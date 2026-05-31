# 🔧 Troubleshooting Guide

All known issues encountered during development and testing, with root causes, fixes, and verification steps.

---

## Table of Contents

1. [Gemini Quota Issue (429 RESOURCE_EXHAUSTED)](#1-gemini-quota-issue-429-resource_exhausted)
2. [dotenv Loading Issue](#2-dotenv-loading-issue)
3. [Invalid Assistant/Model Role Issue](#3-invalid-assistantmodel-role-issue)
4. [Fallback Mode Issue](#4-fallback-mode-issue)
5. [Missing google-generativeai Package](#5-missing-google-generativeai-package)
6. [Pydantic V2 Warnings](#6-pydantic-v2-warnings)
7. [API Key Loading Issues](#7-api-key-loading-issues)
8. [Empty Gemini Response (Safety Filter)](#8-empty-gemini-response-safety-filter)
9. [Conversation History Memory Leak](#9-conversation-history-memory-leak)
10. [CORS Issues in Frontend](#10-cors-issues-in-frontend)

---

## 1. Gemini Quota Issue (429 RESOURCE_EXHAUSTED)

### Problem

The Gemini API returns a `429 RESOURCE_EXHAUSTED` error when the free tier quota is exceeded. This causes the chatbot to fail completely with no response.

### Root Cause

- Google Gemini free tier has rate limits (requests per minute, tokens per minute)
- The primary model (`gemini-2.5-flash`) may hit its quota while other models still have capacity
- The original implementation did not handle quota errors — it just failed

### Fix

Implemented **model fallback chain** in `chatbot_service.py`:

```python
# Fallback models to try if primary model hits quota limits (429)
self._fallback_models = ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash"]

# In chat() method:
for model_name in models_to_try:
    try:
        response = self._call_gemini_with_model(enhanced_message, conv_id, model_name)
        return response
    except Exception as e:
        error_str = str(e)
        is_quota_error = "429" in error_str or "quota" in error_str.lower() or "RESOURCE_EXHAUSTED" in error_str
        if is_quota_error and model_name != models_to_try[-1]:
            logger.warning(f"Model {model_name} hit quota limit (429), trying next fallback model...")
            continue  # Try next model
        else:
            break  # Non-quota error, don't retry
```

### Verification

1. Send multiple rapid chat messages to trigger quota limit
2. Check logs for `[CHAT] Model X hit quota limit (429), trying next fallback model...`
3. Verify that the chatbot still responds (using a fallback model)
4. Check response `model_used` field — it should show the fallback model name

---

## 2. dotenv Loading Issue

### Problem

Environment variables from `.env` file were not being loaded, causing `GEMINI_API_KEY` to be empty even though the `.env` file existed.

### Root Cause

- `load_dotenv()` was called without an explicit path, relying on the current working directory
- When running from different directories (e.g., VS Code terminal, Docker container), the `.env` file path was different
- The default `load_dotenv()` behavior doesn't search parent directories

### Fix

Added explicit path resolution in multiple locations:

```python
# In chatbot_service.py (and chatbot_utils.py):
from pathlib import Path
_env_path = Path(__file__).parent.parent.parent / ".env"
_loaded = load_dotenv(_env_path, override=True)

# In api/main.py:
_env_path = Path(__file__).parent.parent / ".env"
_loaded = load_dotenv(_env_path, override=True)

# In utils/config.py:
_env_path = Path(__file__).parent.parent / ".env"
_loaded = load_dotenv(_env_path, override=True)
```

Key: `override=True` ensures that `.env` values override any existing environment variables.

### Verification

1. Create `.env` file with `GEMINI_API_KEY=your_key`
2. Start the API server from any directory
3. Check logs for `Loaded .env from /path/to/.env`
4. Call `GET /chat/status` and verify `api_key_configured: true`

---

## 3. Invalid Assistant/Model Role Issue

### Problem

Gemini API rejected conversation history with error: `Invalid role: 'assistant'. Role must be 'user' or 'model'.`

### Root Cause

- The chatbot internally stores messages with roles `'user'` and `'assistant'`
- The Gemini API requires roles `'user'` and `'model'` (not `'assistant'`)
- When sending conversation history to Gemini, the `'assistant'` role was not being converted

### Fix

Added role mapping when building Gemini chat history:

```python
# In chatbot_memory.py (build_gemini_history method):
chat_history = [
    {"role": "model" if h["role"] == "assistant" else h["role"], "parts": [h["content"]]}
    for h in history
]
```

Also, Gemini requires the `parts` structure (list of strings), not just a plain string.

### Verification

1. Start a conversation and send 2+ messages with the same `conversation_id`
2. Check logs for `[GEMINI_CALL] Sending message to Gemini API...`
3. Verify no "Invalid role" errors in logs
4. Verify the AI provides context-aware responses (referencing previous messages)

---

## 4. Fallback Mode Issue

### Problem

When Gemini API is unavailable, the chatbot should still provide useful responses, but the original implementation had no fallback mechanism.

### Root Cause

- No pre-defined responses for when Gemini is down
- No topic matching to select relevant fallback responses
- Users would get error messages instead of helpful content

### Fix

Implemented complete fallback system in `chatbot_utils.py`:

1. **FALLBACK_RESPONSES dictionary** — Pre-defined responses for 5 topics:
   - `ic50`: IC50 interpretation
   - `recommendation`: Recommendation engine explanation
   - `preprocessing`: Pipeline description
   - `models`: ML model information
   - `general`: General chatbot introduction

2. **`match_fallback_topic()` function** — Keyword-based topic matching:
   ```python
   def match_fallback_topic(message: str) -> str:
       message_lower = message.lower()
       if any(kw in message_lower for kw in ['ic50', 'ic 50', 'half maximal', ...]):
           return "ic50"
       # ... more topic checks
       return "general"
   ```

3. **Fallback flow in `chat()` method**:
   ```python
   if self.is_available():
       # Try Gemini first
   else:
       # Use fallback response
       fallback_key = match_fallback_topic(message)
       fallback_message = FALLBACK_RESPONSES[fallback_key]
   ```

### Verification

1. Remove or unset `GEMINI_API_KEY`
2. Start the API server
3. Call `GET /chat/status` → verify `available: false`
4. Send `POST /chat {"message": "What does IC50 mean?"}` → verify fallback response with IC50 content
5. Verify `model_used: "fallback"` in response

---

## 5. Missing google-generativeai Package

### Problem

The `google-generativeai` package was not installed, causing `ImportError` when trying to initialize Gemini.

### Root Cause

- The package was not in the initial `requirements.txt`
- Even after adding it, it needed to be explicitly installed with `pip install google-generativeai`

### Fix

1. Added to `requirements.txt`:
   ```
   google-generativeai>=0.3.0
   python-dotenv>=1.0.0
   ```

2. Added graceful handling in `_initialize_gemini()`:
   ```python
   try:
       import google.generativeai as genai
       # ... initialize
   except ImportError:
       logger.warning(
           "google-generativeai package not installed. "
           "Install it with: pip install google-generativeai. "
           "Chatbot will use fallback responses."
       )
       self.api_key_configured = False
   ```

### Verification

1. Uninstall the package: `pip uninstall google-generativeai`
2. Start the API server
3. Check logs for warning message about missing package
4. Verify chatbot still works in fallback mode
5. Reinstall: `pip install google-generativeai>=0.3.0`
6. Verify chatbot switches to Gemini mode

---

## 6. Pydantic V2 Warnings

### Problem

Pydantic V2 deprecated `schema_extra` in the `Config` class, causing warnings:

```
PydanticUserWarning: `schema_extra` is deprecated. Use `json_schema_extra` instead.
```

### Root Cause

- The project uses Pydantic V2 (`pydantic>=1.10.0` allows V2)
- The original schema files used `Config.schema_extra` (V1 syntax)
- Pydantic V2 requires `Config.json_schema_extra`

### Fix

Updated all schema classes in `chatbot_schema.py`:

```python
# BEFORE (Pydantic V1):
class Config:
    schema_extra = {
        "example": { ... }
    }

# AFTER (Pydantic V2):
class Config:
    json_schema_extra = {
        "example": { ... }
    }
```

### Verification

1. Start the API server
2. Check that no PydanticUserWarning appears in logs
3. Visit `/docs` and verify example schemas are shown correctly
4. Visit `/openapi.json` and verify examples are in the schema

---

## 7. API Key Loading Issues

### Problem

The `GEMINI_API_KEY` was not being read correctly from the environment, even after setting it in `.env`.

### Root Cause

Multiple potential causes:
1. `.env` file not in the correct location (project root)
2. `load_dotenv()` called without `override=True` — existing env vars take precedence
3. Placeholder value `your_gemini_api_key_here` not being filtered out
4. `.env` file loaded after the chatbot service was already initialized

### Fix

1. **Explicit path**: Use `Path(__file__).parent.parent / ".env"` to find `.env` relative to the module
2. **Override flag**: Use `load_dotenv(_env_path, override=True)` to ensure `.env` values take precedence
3. **Placeholder check**: Filter out placeholder values:
   ```python
   if not api_key or api_key == "your_gemini_api_key_here":
       logger.warning("GEMINI_API_KEY not configured...")
       self.api_key_configured = False
       return
   ```
4. **Load order**: Load `.env` in `api/main.py` BEFORE importing routers:
   ```python
   # In api/main.py:
   from dotenv import load_dotenv
   _env_path = Path(__file__).parent.parent / ".env"
   load_dotenv(_env_path, override=True)
   
   # THEN import routers
   from api.routers import chatbot
   ```

### Verification

1. Set `GEMINI_API_KEY=your_real_key` in `.env` file in project root
2. Start API server
3. Check logs for `Loaded .env from /path/to/.env`
4. Call `GET /chat/status` → verify `api_key_configured: true`
5. Send a chat message → verify `model_used` is a Gemini model name (not "fallback")

---

## 8. Empty Gemini Response (Safety Filter)

### Problem

Gemini API returns an empty response (no text) when content is filtered by safety settings.

### Root Cause

- Gemini has built-in safety filters that may block certain queries
- When filtered, `response.text` is empty/None
- The `finish_reason` on the candidate indicates why it was filtered

### Fix

Added empty response handling in `_call_gemini()`:

```python
if response.text:
    return response.text
else:
    # Check for safety filter blocks
    try:
        if response.candidates and response.candidates[0].finish_reason:
            logger.warning(f"Candidate finish_reason={response.candidates[0].finish_reason}")
    except Exception:
        pass
    return (
        "I apologize, but I couldn't generate a response for that query. "
        "This may be due to content safety filters. Please try rephrasing your question "
        "or ask about a different topic related to drug response prediction."
    )
```

### Verification

1. Send a message that might trigger safety filters (e.g., asking about treating a real patient)
2. Verify the chatbot returns a polite redirect response instead of crashing
3. Check logs for `Gemini returned empty response (possibly filtered by safety settings)`

---

## 9. Conversation History Memory Leak

### Problem

Long conversations could consume excessive memory if history grows unbounded.

### Root Cause

- No limit on conversation history size
- Each conversation stores all messages indefinitely
- In a production environment with many users, memory usage would grow without bound

### Fix

Added history size limiting in `ConversationMemory`:

```python
MAX_HISTORY_SIZE = 50

def add_message(self, conversation_id, role, content):
    # ... append message
    if len(self.conversations[conversation_id]) > self.max_history:
        self.conversations[conversation_id] = (
            self.conversations[conversation_id][-self.max_history:]
        )
```

### Verification

1. Send 60+ messages in a single conversation
2. Retrieve history via `GET /chat/history/{conversation_id}`
3. Verify that only the most recent 50 messages are returned
4. Check that older messages were trimmed

---

## 10. CORS Issues in Frontend

### Problem

Frontend requests to the chatbot API are blocked by CORS policy.

### Root Cause

- Browser security prevents cross-origin requests by default
- The FastAPI server needs CORS middleware configured

### Fix

CORS middleware is already configured in `api/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Development: allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

For production, restrict origins:

```python
allow_origins=["https://your-production-domain.com"]
```

Or set via environment:

```
CORS_ORIGINS=https://your-app.com
```

### Verification

1. Start the API server
2. Open browser console on your frontend page
3. Send a fetch request to `/chat`
4. Verify no CORS errors in console
5. Verify response is received successfully