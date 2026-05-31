# 🧬 Drug Response Prediction — AI Chatbot Module

A standalone, production-ready AI chatbot module for the **Drug Response Prediction and Recommendation Platform**. This module provides an intelligent assistant that helps users understand IC50 predictions, drug recommendations, preprocessing methodology, and ML model architecture.

> **⚠️ Safety Notice**: This chatbot is for **educational and explainability purposes only**. It does NOT provide medical diagnosis or treatment recommendations.

---

## 📋 Table of Contents

- [What This Module Does](#what-this-module-does)
- [Supported Capabilities](#supported-capabilities)
- [Architecture Overview](#architecture-overview)
- [Gemini Integration](#gemini-integration)
- [Prediction Interpretation](#prediction-interpretation)
- [Recommendation Explanation](#recommendation-explanation)
- [Conversation Memory](#conversation-memory)
- [Fallback Logic](#fallback-logic)
- [Quick Start](#quick-start)
- [File Structure](#file-structure)
- [Environment Setup](#environment-setup)
- [Integration](#integration)

---

## What This Module Does

This chatbot module is an **AI-powered assistant** that provides domain-aware explanations for the Drug Response Prediction system. It:

1. **Explains IC50 values** — What they mean, how they're predicted, and how to interpret them
2. **Interprets predictions** — Given a prediction context (IC50 value, cell line, drug), it provides a specific, tailored explanation
3. **Explains recommendations** — How drugs are ranked, what scores mean, and why certain drugs are prioritized
4. **Describes the preprocessing pipeline** — Label encoding, feature selection, standard scaling
5. **Explains ML models** — CatBoost, XGBoost, LightGBM, and the other 7 regression models
6. **Maintains conversation context** — Users can continue conversations across multiple messages
7. **Operates in fallback mode** — When Gemini API is unavailable, it provides pre-defined domain responses

---

## Supported Capabilities

| Capability | Description | Example Question |
|---|---|---|
| **IC50 Interpretation** | Explains what IC50 values mean and how to interpret sensitivity thresholds | "What does an IC50 of -1.46 mean?" |
| **Prediction Context** | Interprets specific prediction results when context is provided | "Explain this prediction for A172 + Camptothecin" |
| **Recommendation Explanation** | Explains how drugs are ranked and scored | "How does the recommendation engine work?" |
| **Preprocessing Pipeline** | Describes the 3-stage preprocessing (encoding, selection, scaling) | "What preprocessing steps are applied?" |
| **ML Model Information** | Explains the 7 regression models and their differences | "Why is CatBoost the default model?" |
| **Project Architecture** | Describes how prediction and recommendation engines work | "How does the whole system work?" |
| **Data Source Information** | Explains GDSC dataset, cell line details, compound annotations | "What data does the system use?" |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React/Next)                 │
│                                                         │
│  ┌─────────┐  ┌──────────┐  ┌──────────────────────┐   │
│  │ Chat UI │  │ Context  │  │ Conversation Manager  │   │
│  └─────────┘  └──────────┘  └──────────────────────┘   │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP (POST /chat)
                         ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Backend (chatbot_router.py)         │
│                                                         │
│  ┌──────────────────┐  ┌────────────────────────────┐  │
│  │  ChatRequest      │  │  ChatResponse              │  │
│  │  (Pydantic schema)│  │  (Pydantic schema)         │  │
│  └──────────────────┘  └────────────────────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│            ChatbotService (chatbot_service.py)           │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Gemini API   │  │ Fallback     │  │ Conversation  │  │
│  │ Integration  │  │ Responses    │  │ Memory        │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐                    │
│  │ System Prompt│  │ Model        │                    │
│  │ (Domain ctx) │  │ Fallback(429)│                    │
│  └──────────────┘  └──────────────┘                    │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Google Gemini API                           │
│                                                         │
│  Models: gemini-2.5-flash (primary)                     │
│          gemini-2.0-flash-lite (fallback)                │
│          gemini-2.0-flash (fallback)                     │
└─────────────────────────────────────────────────────────┘
```

---

## Gemini Integration

The chatbot uses **Google Gemini** (via `google-generativeai` package) as its AI backend:

- **Primary model**: `gemini-2.5-flash` (configurable via `GEMINI_MODEL` env var)
- **Fallback models**: When the primary model hits quota limits (HTTP 429), the service automatically tries:
  1. `gemini-2.5-flash` → `gemini-2.0-flash-lite` → `gemini-2.0-flash`
- **System instruction**: A domain-specific system prompt is applied that gives the AI full knowledge of the project's architecture, data, and methodology
- **Conversation history**: Previous messages are sent to Gemini as chat history for context continuity
- **Role mapping**: The service maps `assistant` → `model` role for Gemini API compatibility

### Key Implementation Details

```python
# Gemini requires 'model' role (not 'assistant')
chat_history = [
    {"role": "model" if h["role"] == "assistant" else h["role"], "parts": [h["content"]]}
    for h in history
]

# Model fallback on quota errors
for model_name in models_to_try:
    try:
        response = self._call_gemini_with_model(message, conv_id, model_name)
        return response
    except Exception as e:
        if "429" in str(e):  # Quota error — try next model
            continue
        else:
            break  # Non-quota error — don't retry
```

---

## Prediction Interpretation

When a user sends a message with prediction context, the chatbot provides **specific, tailored** interpretations:

### IC50 Sensitivity Thresholds

| IC50 Range | Sensitivity Level | Interpretation |
|---|---|---|
| IC50 < -2 | **Very Sensitive** | Drug is highly effective — needs very low concentration to inhibit cell growth |
| -2 ≤ IC50 < 0 | **Sensitive** | Drug is likely effective for this cell line |
| 0 ≤ IC50 < 2 | **Moderate** | Average effectiveness — further testing may be needed |
| IC50 ≥ 2 | **Resistant** | Drug is likely not effective for this cell line |

### Context Enhancement

When the frontend passes prediction context, the service enhances the message:

```python
# Original user message: "What does this prediction mean?"
# With context: {"predicted_ic50": -1.46, "cell_line": "A172", "drug": "Camptothecin"}

# Enhanced message sent to Gemini:
"[Context: predicted_ic50: -1.46, cell_line: A172, drug: Camptothecin]\n\nUser question: What does this prediction mean?"
```

This ensures the AI provides a **specific interpretation** rather than a generic explanation.

---

## Recommendation Explanation

The chatbot explains the recommendation engine's methodology:

- **Score formula**: `score = 1 / (1 + IC50)`
- **Ranking logic**: Lower IC50 → higher score → better recommendation
- **Output**: Ranked list with drug name, IC50, score, target, and pathway
- **Interpretation**: The top-ranked drugs are those predicted to be most effective for the given cell line

---

## Conversation Memory

The chatbot maintains conversation history using `ConversationMemory`:

- **Storage**: In-memory dictionary keyed by `conversation_id`
- **Max history**: 50 messages per conversation (older messages are trimmed)
- **Message structure**: Each message has `role`, `content`, and `timestamp`
- **Gemini format**: History is converted to Gemini's format (`model` role, `parts` structure) before sending
- **Session continuity**: The frontend must preserve `conversation_id` across messages to maintain context

### How It Works

```
1. User sends first message → new conversation_id is generated (UUID)
2. Response includes conversation_id → frontend stores it
3. User sends next message with same conversation_id → history is loaded
4. Gemini receives full conversation history → provides context-aware response
5. New messages are appended to history → up to 50 messages per conversation
```

---

## Fallback Logic

When the Gemini API is unavailable, the chatbot operates in **fallback mode**:

1. **Topic matching**: The user's message is analyzed for keywords to determine the topic
2. **Pre-defined responses**: A domain-specific fallback response is returned for the matched topic
3. **Topics covered**: IC50, recommendations, preprocessing, models, general
4. **User notification**: Fallback responses include a note that Gemini AI is unavailable
5. **Configuration advice**: Users are advised to configure `GEMINI_API_KEY` for full AI responses

### Fallback Trigger Conditions

- `GEMINI_API_KEY` not set or set to placeholder value
- `google-generativeai` package not installed
- Gemini API initialization fails
- All model fallbacks exhausted (quota errors on all models)
- Non-quota API errors

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r chatbot_module/REQUIREMENTS.md  # Or install individually:
pip install google-generativeai>=0.3.0 python-dotenv>=1.0.0 fastapi>=0.95.0 uvicorn>=0.21.0 pydantic>=1.10.0
```

### 2. Configure Environment

```bash
cp chatbot_module/.env.example .env
# Edit .env and add your GEMINI_API_KEY
```

Get your API key from: https://aistudio.google.com/app/apikey

### 3. Integrate into FastAPI

```python
# In your main.py:
from chatbot_module.backend.chatbot_router import router as chatbot_router

app = FastAPI(...)
app.include_router(chatbot_router)
```

### 4. Test the Chatbot

```bash
# Start your API server
uvicorn api.main:app --reload

# Check chatbot status
curl http://localhost:8000/chat/status

# Send a message
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What does IC50 mean?"}'
```

---

## File Structure

```
chatbot_module/
│
├── README.md                          ← This file
├── INTEGRATION_GUIDE.md               ← Frontend integration instructions
├── API_REFERENCE.md                   ← Full API documentation
├── TROUBLESHOOTING.md                 ← Known issues and fixes
├── REQUIREMENTS.md                    ← Python dependencies
├── .env.example                       ← Environment variable template
│
├── backend/
│   ├── __init__.py                    ← Package exports
│   ├── chatbot_service.py             ← Core service (Gemini, fallback, singleton)
│   ├── chatbot_router.py              ← FastAPI router (endpoints)
│   ├── chatbot_schema.py              ← Pydantic request/response models
│   ├── chatbot_memory.py              ← Conversation history management
│   └── chatbot_utils.py               ← Env loading, fallback responses, topic matching
│
├── prompts/
│   ├── system_prompt.txt              ← Full domain-specific system prompt
│   ├── safety_rules.txt               ← Safety restrictions (mandatory)
│   └── chatbot_behavior.txt           ← Communication style and response format
│
├── frontend_integration/
│   ├── API_EXAMPLES.md                ← fetch/axios/React examples
│   ├── REACT_INTEGRATION.md           ← React component integration guide
│   ├── CHAT_FLOW.md                   ← Chat flow diagram and sequence
│   └── SAMPLE_REQUESTS.json           ← Sample JSON payloads for testing
│
└── tests/
    ├── test_chatbot.py                ← Chatbot service unit tests
    ├── test_gemini.py                 ← Gemini integration tests
    └── test_api.py                    ← API endpoint tests
```

---

## Environment Setup

See [`.env.example`](.env.example) for the full template.

Required variables:

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | **Yes** | — | Google Gemini API key (get from https://aistudio.google.com/app/apikey) |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Gemini model to use |

Optional variables (for the full API server):

| Variable | Default | Description |
|---|---|---|
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |
| `API_RELOAD` | `True` | Enable auto-reload for development |
| `DEFAULT_MODEL` | `CatBoost` | Default ML prediction model |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |

---

## Integration

- **Backend integration**: See [`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md)
- **Frontend integration**: See [`frontend_integration/REACT_INTEGRATION.md`](frontend_integration/REACT_INTEGRATION.md)
- **API details**: See [`API_REFERENCE.md`](API_REFERENCE.md)
- **Troubleshooting**: See [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)