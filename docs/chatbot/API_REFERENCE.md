# 📡 API Reference — Chatbot Endpoints

Complete documentation for all chatbot API endpoints, including request/response formats and error handling.

---

## Base URL

```
http://localhost:8000/chat
```

---

## Endpoints

### 1. POST /chat — Send a Message

Send a message to the AI chatbot and receive a domain-aware response.

**Request Body**: `ChatRequest`

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `message` | string | **Yes** | 1–2000 chars, non-empty after trim | User's question or message |
| `conversation_id` | string | No | UUID format | ID to continue a previous conversation |
| `context` | object | No | Any key-value pairs | Prediction data for specific interpretation |

#### Request Examples

**Basic message (new conversation):**

```json
POST /chat
Content-Type: application/json

{
  "message": "What does IC50 mean?"
}
```

**Continue existing conversation:**

```json
POST /chat
Content-Type: application/json

{
  "message": "Can you explain more about the preprocessing pipeline?",
  "conversation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

**Message with prediction context:**

```json
POST /chat
Content-Type: application/json

{
  "message": "What does this prediction mean?",
  "conversation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "context": {
    "predicted_ic50": -1.46,
    "model_used": "CatBoost",
    "cell_line_name": "A172",
    "drug_name": "Camptothecin",
    "target": "TOP1",
    "target_pathway": "DNA replication"
  }
}
```

**Message with recommendation context:**

```json
POST /chat
Content-Type: application/json

{
  "message": "Why is Camptothecin ranked as the top drug?",
  "context": {
    "cell_line_name": "A172",
    "top_drug": "Camptothecin",
    "top_drug_ic50": -1.46,
    "top_drug_score": 0.87,
    "total_drugs_evaluated": 265
  }
}
```

#### Response: `ChatResponse`

| Field | Type | Description |
|---|---|---|
| `success` | boolean | Whether the response was generated successfully |
| `message` | string | AI assistant response (may contain markdown formatting) |
| `conversation_id` | string | Conversation UUID — store this for future messages |
| `timestamp` | string | ISO 8601 timestamp of the response |
| `model_used` | string | Model that generated the response (`"gemini-2.5-flash"` or `"fallback"`) |

#### Success Response Example

```json
{
  "success": true,
  "message": "An IC50 value of -1.46 indicates that the drug Camptothecin is highly effective for the A172 cell line. This falls in the 'Sensitive' range (IC50 between -2 and 0), meaning the drug requires a relatively low concentration to inhibit 50% of cell growth.\n\n**Key factors the model considered:**\n- The target TOP1 (Topoisomerase I) — a well-characterized cancer target\n- The DNA replication pathway — critical for cell proliferation\n- The GBM (Glioblastoma) tissue type\n\n⚠️ **Reminder**: This prediction is for educational and research purposes only. It should not be used to make clinical treatment decisions.",
  "conversation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2026-05-25T12:00:00",
  "model_used": "gemini-2.5-flash"
}
```

#### Fallback Response Example (Gemini Unavailable)

```json
{
  "success": true,
  "message": "IC50 (Half Maximal Inhibitory Concentration) measures the concentration of a drug needed to inhibit 50% of cell growth. Lower IC50 values indicate higher drug effectiveness. In our system, we predict LN_IC50 (natural log of IC50):\n- IC50 < -2: Very sensitive (drug highly effective)\n- IC50 between -2 and 0: Sensitive (drug likely effective)\n- IC50 between 0 and 2: Moderate effectiveness\n- IC50 > 2: Resistant (drug likely not effective)\n\n⚠️ Note: The Gemini AI service is currently unavailable. This is a pre-defined response. For more detailed explanations, please ensure the GEMINI_API_KEY is configured.",
  "conversation_id": "f0e1d2c3-b4a5-6789-0abc-def123456789",
  "timestamp": "2026-05-25T12:00:00",
  "model_used": "fallback"
}
```

#### Error Responses

**Validation Error (422)** — Empty message:

```json
{
  "success": false,
  "error": "Validation error",
  "detail": [
    {
      "loc": ["body", "message"],
      "msg": "Message cannot be empty or contain only whitespace",
      "type": "value_error"
    }
  ],
  "body": {
    "message": "   ",
    "conversation_id": null,
    "context": null
  }
}
```

**Validation Error (422)** — Message too long:

```json
{
  "success": false,
  "error": "Validation error",
  "detail": [
    {
      "loc": ["body", "message"],
      "msg": "String should have at most 2000 characters",
      "type": "string_too_long"
    }
  ]
}
```

**Server Error (500)** — Gemini API failure:

```json
{
  "success": false,
  "error": "Internal server error",
  "detail": "Chatbot error: Gemini API call failed with gemini-2.5-flash: RESOURCE_EXHAUSTED"
}
```

---

### 2. GET /chat/status — Check Chatbot Status

Check if the chatbot service is available and properly configured.

**Request**: No parameters

```json
GET /chat/status
```

#### Response: `ChatbotStatusResponse`

| Field | Type | Description |
|---|---|---|
| `available` | boolean | Whether the chatbot has Gemini API access |
| `model` | string or null | Gemini model name (null if unavailable) |
| `api_key_configured` | boolean | Whether GEMINI_API_KEY is set |
| `message` | string | Human-readable status description |

#### Available Response Example

```json
{
  "available": true,
  "model": "gemini-2.5-flash",
  "api_key_configured": true,
  "message": "Chatbot service is available with Gemini AI"
}
```

#### Fallback Mode Response Example

```json
{
  "available": false,
  "model": null,
  "api_key_configured": false,
  "message": "Chatbot service is in fallback mode (GEMINI_API_KEY not configured)"
}
```

---

### 3. GET /chat/history/{conversation_id} — Get Conversation History

Retrieve all messages in a specific conversation.

**Path Parameter**:

| Parameter | Type | Description |
|---|---|---|
| `conversation_id` | string | The conversation UUID |

```json
GET /chat/history/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

#### Response: `ChatHistoryResponse`

| Field | Type | Description |
|---|---|---|
| `success` | boolean | Whether history retrieval was successful |
| `conversation_id` | string | The conversation ID |
| `messages` | array | List of ChatMessage objects |
| `timestamp` | string | ISO 8601 timestamp |

#### ChatMessage Object

| Field | Type | Description |
|---|---|---|
| `role` | string | `"user"` or `"assistant"` |
| `content` | string | Message text |
| `timestamp` | string | ISO 8601 timestamp |

#### Success Response Example

```json
{
  "success": true,
  "conversation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "messages": [
    {
      "role": "user",
      "content": "What does IC50 mean?",
      "timestamp": "2026-05-25T11:55:00"
    },
    {
      "role": "assistant",
      "content": "IC50 (Half Maximal Inhibitory Concentration) measures...",
      "timestamp": "2026-05-25T11:55:05"
    },
    {
      "role": "user",
      "content": "Can you explain the preprocessing pipeline?",
      "timestamp": "2026-05-25T11:56:00"
    },
    {
      "role": "assistant",
      "content": "The preprocessing pipeline has 3 stages...",
      "timestamp": "2026-05-25T11:56:08"
    }
  ],
  "timestamp": "2026-05-25T12:00:00"
}
```

#### Empty History Response (Unknown conversation_id)

```json
{
  "success": true,
  "conversation_id": "unknown-id-12345",
  "messages": [],
  "timestamp": "2026-05-25T12:00:00"
}
```

---

### 4. DELETE /chat/history/{conversation_id} — Clear Conversation History

Clear all messages for a specific conversation.

**Path Parameter**:

| Parameter | Type | Description |
|---|---|---|
| `conversation_id` | string | The conversation UUID to clear |

```json
DELETE /chat/history/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

#### Response Examples

**Existing conversation cleared:**

```json
{
  "success": true,
  "message": "Conversation a1b2c3d4-e5f6-7890-abcd-ef1234567890 cleared"
}
```

**Unknown conversation_id:**

```json
{
  "success": true,
  "message": "Conversation unknown-id-12345 not found (no action needed)"
}
```

---

## Error Handling Summary

| HTTP Status | When | Response Format |
|---|---|---|
| `200` | Successful response | `ChatResponse` / `ChatHistoryResponse` / `ChatbotStatusResponse` |
| `422` | Validation error (empty message, too long, invalid role) | `{success: false, error: "Validation error", detail: [...]}` |
| `500` | Server error (Gemini failure, internal error) | `{success: false, error: "Internal server error", detail: "..."}` |

---

## Rate Limits and Quotas

- **Gemini API**: Subject to Google's rate limits. The service automatically falls back to alternative models on 429 (quota exceeded) errors.
- **Fallback models**: `gemini-2.5-flash` → `gemini-2.0-flash-lite` → `gemini-2.0-flash`
- **Conversation memory**: Limited to 50 messages per conversation. Older messages are automatically trimmed.
- **Message length**: Maximum 2000 characters per message (enforced by Pydantic validation).

---

## Swagger/OpenAPI Documentation

When the FastAPI server is running, interactive API documentation is available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`