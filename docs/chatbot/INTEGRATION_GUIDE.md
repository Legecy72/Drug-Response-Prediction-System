# 🔗 Chatbot Integration Guide

This guide explains how to integrate the AI chatbot module into your frontend application. It is written for **frontend developers** who will build the production UI.

> **Important**: The current demo frontend (`web/index.html`) was only a temporary testing interface. The production frontend should be built separately using this guide.

---

## Table of Contents

- [Overview](#overview)
- [Backend Connection](#backend-connection)
- [Sending Messages](#sending-messages)
- [Passing Prediction Context](#passing-prediction-context)
- [Handling Responses](#handling-responses)
- [Preserving Conversation ID](#preserving-conversation-id)
- [Rendering AI Responses](#rendering-ai-responses)
- [Chat Flow Sequence](#chat-flow-sequence)
- [Error Handling](#error-handling)
- [CORS Configuration](#cors-configuration)

---

## Overview

The chatbot is a **backend service** accessed via REST API endpoints. Your frontend sends HTTP requests to the FastAPI backend, and receives structured JSON responses.

**Key principle**: The chatbot is **decoupled** from any specific UI framework. You can integrate it with React, Vue, Angular, or any other framework — the API is the same.

---

## Backend Connection

### Base URL

```
http://localhost:8000/chat
```

In production, replace `localhost:8000` with your deployed API URL.

### Required Headers

```json
{
  "Content-Type": "application/json"
}
```

### CORS

The backend is configured with `allow_origins=["*"]` for development. In production, restrict this to your frontend domain:

```python
# In api/main.py or your FastAPI app:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-production-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Sending Messages

### Basic Message (No Context)

```json
POST /chat
{
  "message": "What does IC50 mean?"
}
```

### Message with Conversation ID (Continue Previous Chat)

```json
POST /chat
{
  "message": "Can you explain more about that?",
  "conversation_id": "conv_abc123"
}
```

### Message with Prediction Context

```json
POST /chat
{
  "message": "What does this prediction mean?",
  "conversation_id": "conv_abc123",
  "context": {
    "predicted_ic50": -1.46,
    "model_used": "CatBoost",
    "cell_line_name": "A172",
    "drug_name": "Camptothecin"
  }
}
```

### Field Specifications

| Field | Type | Required | Max Length | Description |
|---|---|---|---|---|
| `message` | string | **Yes** | 2000 chars | User's question or message |
| `conversation_id` | string | No | — | UUID for conversation continuity |
| `context` | object | No | — | Prediction data for specific interpretation |

---

## Passing Prediction Context

The `context` field is the **most powerful feature** of this chatbot. It allows the AI to provide **specific, tailored** interpretations instead of generic explanations.

### When to Pass Context

Pass context whenever the user is asking about a **specific prediction result**:

- After a prediction is made (user clicks "Explain this prediction")
- When viewing a recommendation table (user clicks "Why is this drug ranked #1?")
- When comparing models (user asks about a specific IC50 value)

### What to Include in Context

You can include any key-value pairs. The most useful ones:

```json
{
  "predicted_ic50": -1.46,       // The predicted IC50 value
  "model_used": "CatBoost",      // Which ML model was used
  "cell_line_name": "A172",      // The cell line name
  "drug_name": "Camptothecin",   // The drug name
  "target": "TOP1",              // Drug target
  "target_pathway": "DNA replication",  // Target pathway
  "sensitivity": "Sensitive"     // Your interpretation label
}
```

### How Context Is Processed

The backend transforms context into a structured prefix:

```
[Context: predicted_ic50: -1.46, model_used: CatBoost, cell_line_name: A172, drug_name: Camptothecin]

User question: What does this prediction mean?
```

This is sent to Gemini, which then provides a **specific interpretation** of that exact prediction.

---

## Handling Responses

### Success Response

```json
{
  "success": true,
  "message": "An IC50 value of -1.46 indicates that the drug Camptothecin is highly effective for the A172 cell line. This falls in the 'Sensitive' range (IC50 between -2 and 0), meaning the drug requires a relatively low concentration to inhibit 50% of cell growth.\n\n**Key factors the model considered:**\n- The target TOP1 (Topoisomerase I) — a well-characterized cancer target\n- The DNA replication pathway — critical for cell proliferation\n- The GBM (Glioblastoma) tissue type\n\n⚠️ **Reminder**: This prediction is for educational and research purposes only. It should not be used to make clinical treatment decisions.",
  "conversation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2026-05-25T12:00:00",
  "model_used": "gemini-2.5-flash"
}
```

### Fallback Response (Gemini Unavailable)

```json
{
  "success": true,
  "message": "IC50 (Half Maximal Inhibitory Concentration) measures the concentration of a drug needed to inhibit 50% of cell growth...\n\n⚠️ Note: The Gemini AI service is currently unavailable...",
  "conversation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2026-05-25T12:00:00",
  "model_used": "fallback"
}
```

### Error Response

```json
{
  "success": false,
  "error": "Chatbot error",
  "detail": "Gemini API call failed: quota exceeded"
}
```

### Key Response Fields

| Field | Type | Description |
|---|---|---|
| `success` | boolean | Always `true` for successful responses (even fallback) |
| `message` | string | The AI response text (may contain markdown formatting) |
| `conversation_id` | string | UUID — **store this in frontend state** for continuity |
| `timestamp` | string | ISO 8601 timestamp |
| `model_used` | string | `"gemini-2.5-flash"` for Gemini, `"fallback"` for pre-defined responses |

---

## Preserving Conversation ID

**This is critical for conversation continuity.**

### How It Works

1. **First message**: Don't send `conversation_id` → backend generates a new UUID → returns it in the response
2. **Subsequent messages**: Send the `conversation_id` from the previous response → backend loads history → AI has context

### Frontend State Management

```javascript
// React example
const [conversationId, setConversationId] = useState(null);

const sendMessage = async (message, context = null) => {
  const response = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      conversation_id: conversationId,  // null for first message, UUID for subsequent
      context
    })
  });
  
  const data = await response.json();
  
  // CRITICAL: Store the conversation_id for future messages
  if (data.conversation_id) {
    setConversationId(data.conversation_id);
  }
  
  return data;
};
```

### When to Reset Conversation ID

- User explicitly starts a "new conversation" (clear chat button)
- User switches to a completely different topic
- Conversation becomes too long (backend limits to 50 messages)

### Clearing Conversation

```json
DELETE /chat/history/{conversation_id}
```

---

## Rendering AI Responses

The AI responses may contain **markdown formatting**:

- **Bold text**: `**key factors**`
- **Bullet points**: `- item 1\n- item 2`
- **Headers**: `## Section Title`
- **Warning symbols**: `⚠️`

### Recommended Rendering Approach

Use a markdown renderer in your frontend:

```javascript
// React: use react-markdown
import ReactMarkdown from 'react-markdown';

function ChatMessage({ message }) {
  return <ReactMarkdown>{message}</ReactMarkdown>;
}
```

### Styling Suggestions

- Render `⚠️` warnings with a distinct color (e.g., orange/yellow)
- Bold key terms for emphasis
- Use monospace for technical values (IC50, scores)
- Add a subtle background for AI messages vs user messages

---

## Chat Flow Sequence

```
┌──────────┐                              ┌──────────┐                    ┌──────────┐
│  Frontend │                              │  Backend  │                    │  Gemini  │
└──────────┘                              └──────────┘                    └──────────┘
     │                                         │                              │
     │  POST /chat {message}                   │                              │
     │─────────────────────────────────────────▶│                              │
     │                                         │  No conversation_id?          │
     │                                         │  Generate UUID                │
     │                                         │                              │
     │                                         │  Build enhanced message       │
     │                                         │  (add context if provided)    │
     │                                         │                              │
     │                                         │  Send to Gemini API           │
     │                                         │──────────────────────────────▶│
     │                                         │                              │
     │                                         │  Gemini response              │
     │                                         │◀──────────────────────────────│
     │                                         │                              │
     │                                         │  Store in conversation memory │
     │                                         │                              │
     │  Response {message, conversation_id}    │                              │
     │◀─────────────────────────────────────────│                              │
     │                                         │                              │
     │  Frontend stores conversation_id        │                              │
     │                                         │                              │
     │  POST /chat {message, conversation_id}  │                              │
     │─────────────────────────────────────────▶│                              │
     │                                         │  Load history from memory     │
     │                                         │                              │
     │                                         │  Send message + history       │
     │                                         │──────────────────────────────▶│
     │                                         │                              │
     │                                         │  Gemini context-aware response│
     │                                         │◀──────────────────────────────│
     │                                         │                              │
     │  Response with context-aware answer     │                              │
     │◀─────────────────────────────────────────│                              │
```

---

## Error Handling

### Network Errors

```javascript
try {
  const response = await fetch('/chat', { ... });
  if (!response.ok) {
    const errorData = await response.json();
    // Handle HTTP errors (500, 422, etc.)
    showError(errorData.detail || `Error: ${response.status}`);
  }
  const data = await response.json();
  // Process successful response
} catch (error) {
  // Handle network errors (server down, CORS, etc.)
  showError('Unable to connect to the chatbot service.');
}
```

### Validation Errors (422)

If the message is empty or exceeds 2000 characters:

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
  ]
}
```

### Fallback Mode Detection

Check `model_used` field:

```javascript
if (data.model_used === "fallback") {
  // Show a notice: "AI responses are limited. Configure GEMINI_API_KEY for full responses."
}
```

### Status Check

```javascript
// Check if chatbot is available before showing chat UI
const statusResponse = await fetch('/chat/status');
const status = await statusResponse.json();

if (!status.available) {
  // Show notice: "Chatbot is in fallback mode. Responses will be limited."
}
```

---

## CORS Configuration

For local development, the backend allows all origins (`*`). For production:

1. Update `CORS_ORIGINS` in `.env`:
   ```
   CORS_ORIGINS=https://your-app.com,https://staging.your-app.com
   ```

2. Or update the FastAPI middleware directly:
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://your-app.com"],
       ...
   )
   ```

---

## Next Steps

- See [`API_REFERENCE.md`](API_REFERENCE.md) for complete endpoint documentation
- See [`frontend_integration/API_EXAMPLES.md`](frontend_integration/API_EXAMPLES.md) for code examples
- See [`frontend_integration/REACT_INTEGRATION.md`](frontend_integration/REACT_INTEGRATION.md) for React-specific guide
- See [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for known issues and fixes