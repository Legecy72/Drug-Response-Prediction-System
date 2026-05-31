# 📡 API Examples — Chatbot Integration

Complete code examples for sending messages, retrieving history, and checking status
using **fetch**, **axios**, and **React**.

---

## Table of Contents

- [Fetch API Examples](#fetch-api-examples)
- [Axios Examples](#axios-examples)
- [React Integration Examples](#react-integration-examples)
- [Sample JSON Payloads](#sample-json-payloads)
- [Error Handling](#error-handling)
- [WebSocket Alternative](#websocket-alternative)
- [TypeScript SDK Alternative](#typescript-sdk-alternative)
- [Python SDK Alternative](#python-sdk-alternative)
- [Conversation Management](#conversation-management)
- [Context Enhancement](#context-enhancement)
- [Status Checking](#status-checking)
- [History Retrieval](#history-retrieval)
- [Conversation Clearing](#conversation-clearing)
- [Streaming Responses](#streaming-responses)
- [Polling Pattern](#polling-pattern)
- [Batch Requests](#batch-requests)
- [Error Handling](#error-handling-1)
- [Retry Logic](#retry-logic)
- [Timeout Configuration](#timeout-configuration)
- [Loading States](#loading-states)
- [Environment Variables](#environment-variables)
- [Type Definitions](#type-definitions)
- [Utility Functions](#utility-functions)
- [Complete API Reference](#complete-api-reference)
- [Integration Guide](#integration-guide)
- [Troubleshooting Guide](#troubleshooting-guide)
- [README](#readme)
- [System Prompt](#system-prompt)
- [Safety Rules](#safety-rules)
- [Chatbot Behavior](#chatbot-behavior)
- [Backend Architecture](#backend-architecture)
- [Chatbot Service](#chatbot-service)
- [Chatbot Router](#chatbot-router)
- [Chatbot Schema](#chatbot-schema)
- [Chatbot Memory](#chatbot-memory)
- [Chatbot Utils](#chatbot-utils)
- [File Structure](#file-structure)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Integration](#integration)
- [License](#license)
- [Changelog](#changelog)
- [Contributing](#contributing)
- [Code of Conduct](#code-of-conduct)
- [Security](#security)
- [Acknowledgments](#acknowledgments)
- [Notes on Contributions](#notes-on-contributions)
- [Third-Party Licenses](#third-party-licenses)
- [Additional Credits](#additional-credits)
- [Contact](#contact)

---

## Fetch API Examples

### Basic Message (New Conversation)

```javascript
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: 'What does IC50 mean?'
  })
});

const data = await response.json();
console.log(data);
// {
//   success: true,
//   message: "IC50 (Half Maximal Inhibitory Concentration) measures...",
//   conversation_id: "a1b2c3d4-e5f6-...",
//   model_used: "gemini-2.5-flash"
// }
```

### Message with Prediction Context

```javascript
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: 'What does this prediction mean?',
    conversation_id: 'conv_abc123',
    context: {
      predicted_ic50: -1.46,
      model_used: 'CatBoost',
      cell_line_name: 'A172',
      drug_name: 'Camptothecin'
    }
  })
});

const data = await response.json();
console.log(data.message);
// "An IC50 value of -1.46 indicates that the drug Camptothecin..."
```

### Continue Conversation (Same conversation_id)

```javascript
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: 'Can you explain more about the preprocessing pipeline?',
    conversation_id: data.conversation_id  // Use the ID from previous response
  })
});
```

### Check Chatbot Status

```javascript
const response = await fetch('http://localhost:8000/chat/status');
const status = await response.json();
console.log(status);
// { available: true, model: "gemini-2.5-flash", api_key_configured: true, message: "..." }
```

### Get Conversation History

```javascript
const response = await fetch(`http://localhost:8000/chat/history/${conversationId}`);
const history = await response.json();
console.log(history.messages);
```

### Clear Conversation History

```javascript
const response = await fetch(`http://localhost:8000/chat/history/${conversationId}`, {
  method: 'DELETE'
});
const result = await response.json();
console.log(result);
// { success: true, message: "Conversation conv_abc123 cleared" }
```

---

## Axios Examples

### Install Axios

```bash
npm install axios
```

### Basic Message

```javascript
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

const response = await axios.post(`${API_BASE}/chat`, {
  message: 'What does IC50 mean?'
});

console.log(response.data);
```

### Message with Context

```javascript
const response = await axios.post(`${API_BASE}/chat`, {
  message: 'What does this prediction mean?',
  conversation_id: 'conv_abc123',
  context: {
    predicted_ic50: -1.46,
    model_used: 'CatBoost',
    cell_line_name: 'A172',
    drug_name: 'Camptothecin'
  }
});
```

### Check Status

```javascript
const response = await axios.get(`${API_BASE}/chat/status`);
console.log(response.data.available);
```

### Get History

```javascript
const response = await axios.get(`${API_BASE}/chat/history/${conversationId}`);
console.log(response.data.messages);
```

### Clear History

```javascript
const response = await axios.delete(`${API_BASE}/chat/history/${conversationId}`);
console.log(response.data);
```

### Axios with Error Handling

```javascript
try {
  const response = await axios.post(`${API_BASE}/chat`, {
    message: 'What does IC50 mean?'
  });
  console.log(response.data);
} catch (error) {
  if (error.response) {
    // Server responded with error status
    console.error('Server error:', error.response.data);
  } else if (error.request) {
    // Request was made but no response received
    console.error('Network error: No response received');
  } else {
    // Something happened in setting up the request
    console.error('Error:', error.message);
  }
}
```

---

## React Integration Examples

### Basic Chat Component

```jsx
import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';

const API_BASE = 'http://localhost:8000';

function ChatComponent() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [conversationId, setConversationId] = useState(null);
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: input,
          conversation_id: conversationId
        })
      });

      const data = await response.json();

      if (data.conversation_id) {
        setConversationId(data.conversation_id);
      }

      const assistantMessage = { role: 'assistant', content: data.message };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage = { role: 'assistant', content: 'Error: Unable to connect to chatbot.' };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`chat-message ${msg.role}`}>
            <ReactMarkdown>{msg.content}</ReactMarkdown>
          </div>
        ))}
      </div>
      <div className="chat-input">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about IC50, predictions, recommendations..."
          disabled={loading}
          maxLength={2000}
        />
        <button onClick={sendMessage} disabled={loading || !input.trim()}>
          {loading ? 'Sending...' : 'Send'}
        </button>
      </div>
    </div>
  );
}
```

### Chat with Prediction Context

```jsx
function PredictionChatComponent({ predictionResult }) {
  const [conversationId, setConversationId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessageWithContext = async () => {
    if (!input.trim()) return;

    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: input,
          conversation_id: conversationId,
          context: {
            predicted_ic50: predictionResult.predicted_ic50,
            model_used: predictionResult.model_used,
            cell_line_name: predictionResult.cell_line_name,
            drug_name: predictionResult.drug_name
          }
        })
      });

      const data = await response.json();

      setConversationId(data.conversation_id);
      setMessages(prev => [
        ...prev,
        { role: 'user', content: input },
        { role: 'assistant', content: data.message }
      ]);
      setInput('');
    } catch (error) {
      console.error('Chat error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="prediction-chat">
      <button onClick={sendMessageWithContext} disabled={loading}>
        💬 Explain this prediction
      </button>
      {/* ... rest of chat UI */}
    </div>
  );
}
```

---

## Sample JSON Payloads

See [`SAMPLE_REQUESTS.json`](SAMPLE_REQUESTS.json) for complete sample request/response payloads.

### Quick Reference

| Endpoint | Method | Payload |
|---|---|---|
| `/chat` | POST | `{"message": "What does IC50 mean?"}` |
| `/chat` | POST | `{"message": "...", "conversation_id": "...", "context": {...}}` |
| `/chat/status` | GET | (no body) |
| `/chat/history/{id}` | GET | (no body) |
| `/chat/history/{id}` | DELETE | (no body) |

---

## Error Handling

```javascript
async function safeChatRequest(message, conversationId = null, context = null) {
  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, conversation_id: conversationId, context })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      // Network error — server unreachable
      return {
        success: false,
        message: 'Unable to connect to the chatbot service. Please check your network connection.',
        model_used: 'error'
      };
    }
    throw error;
  }
}
```

---

## Conversation Management

```javascript
class ChatSession {
  constructor(apiBase = 'http://localhost:8000') {
    this.apiBase = apiBase;
    this.conversationId = null;
    this.messages = [];
  }

  async send(message, context = null) {
    const response = await fetch(`${this.apiBase}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        conversation_id: this.conversationId,
        context
      })
    });

    const data = await response.json();

    if (data.conversation_id) {
      this.conversationId = data.conversation_id;
    }

    this.messages.push({ role: 'user', content: message });
    this.messages.push({ role: 'assistant', content: data.message });

    return data;
  }

  async getHistory() {
    if (!this.conversationId) return { messages: [] };

    const response = await fetch(
      `${this.apiBase}/chat/history/${this.conversationId}`
    );
    return await response.json();
  }

  async clear() {
    if (!this.conversationId) return;

    await fetch(
      `${this.apiBase}/chat/history/${this.conversationId}`,
      { method: 'DELETE' }
    );

    this.conversationId = null;
    this.messages = [];
  }

  isFallbackMode(data) {
    return data.model_used === 'fallback';
  }
}
```

---

## Type Definitions (TypeScript)

```typescript
interface ChatRequest {
  message: string;
  conversation_id?: string | null;
  context?: Record<string, any> | null;
}

interface ChatResponse {
  success: boolean;
  message: string;
  conversation_id: string | null;
  timestamp: string;
  model_used: string | null;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

interface ChatHistoryResponse {
  success: boolean;
  conversation_id: string | null;
  messages: ChatMessage[];
  timestamp: string;
}

interface ChatbotStatusResponse {
  available: boolean;
  model: string | null;
  api_key_configured: boolean;
  message: string;
}