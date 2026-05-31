# ⚛️ React Integration Guide

Step-by-step guide for integrating the chatbot module into a React application.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Basic Chat Component](#basic-chat-component)
- [Chat with Prediction Context](#chat-with-prediction-context)
- [Conversation Management](#conversation-management)
- [State Management](#state-management)
- [Styling and UI](#styling-and-ui)
- [Error Handling](#error-handling)
- [Testing](#testing)
- [Production Considerations](#production-considerations)
- [WebSocket Alternative](#websocket-alternative)
- [TypeScript Alternative](#typescript-alternative)
- [Next.js Alternative](#nextjs-alternative)
- [Vue.js Alternative](#vuejs-alternative)
- [Angular Alternative](#angular-alternative)
- [Svelte Alternative](#svelte-alternative)
- [Full API Reference](#full-api-reference)
- [Troubleshooting](#troubleshooting)
- [Changelog](#changelog)
- [Contributing](#contributing)
- [License](#license)
- [Code of Conduct](#code-of-conduct)
- [Security](#security)
- [Acknowledgments](#acknowledgments)
- [Notes on Contributions](#notes-on-contributions)
- [Third-Party Licenses](#third-party-licenses)
- [Additional Credits](#additional-credits)
- [Contact](#contact)

---

## Overview

The chatbot module is designed to be **framework-agnostic** — it works with any frontend technology via REST API. This guide provides React-specific integration patterns, but the same API calls work with Vue, Angular, Svelte, or any other framework.

**Key principle**: The chatbot is **decoupled** from the UI. You send HTTP requests to `/chat` and receive structured JSON responses.

---

## Prerequisites

- React 18+ (or React 17 with hooks)
- `react-markdown` for rendering AI responses (optional but recommended)
- `axios` or native `fetch` for HTTP requests

```bash
npm install react-markdown
# Optional:
npm install axios
```

---

## Installation

### 1. Add the Chatbot Router to Your Backend

```python
# In your FastAPI main.py:
from chatbot_module.backend.chatbot_router import router as chatbot_router

app.include_router(chatbot_router)
```

### 2. Configure Environment

```bash
# In your .env file:
GEMINI_API_KEY=your_actual_api_key
GEMINI_MODEL=gemini-2.5-flash
```

### 3. Create the Chat Component

Copy the component code from this guide into your React project.

---

## Basic Chat Component

```jsx
import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';

const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:8000';

function ChatBot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [conversationId, setConversationId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [chatbotStatus, setChatbotStatus] = useState(null);
  const messagesEndRef = useRef(null);

  // Check chatbot status on mount
  useEffect(() => {
    checkStatus();
  }, []);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const checkStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/chat/status`);
      const data = await response.json();
      setChatbotStatus(data);
    } catch (error) {
      setChatbotStatus({ available: false, message: 'Unable to check status' });
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage = { role: 'user', content: input.trim() };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: input.trim(),
          conversation_id: conversationId
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();

      // CRITICAL: Store conversation_id for continuity
      if (data.conversation_id) {
        setConversationId(data.conversation_id);
      }

      const assistantMessage = { role: 'assistant', content: data.message };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: `⚠️ Error: ${error.message}. Please try again.`
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const startNewConversation = () => {
    setMessages([]);
    setConversationId(null);
    setInput('');
  };

  return (
    <div className="chatbot-container">
      {/* Header */}
      <div className="chatbot-header">
        <h3>🧬 Drug Response AI Assistant</h3>
        {chatbotStatus && (
          <span className={`status-badge ${chatbotStatus.available ? 'available' : 'fallback'}`}>
            {chatbotStatus.available ? '🟢 Gemini AI Active' : '⚠️ Fallback Mode'}
          </span>
        )}
        <button onClick={startNewConversation} className="new-chat-btn">
          🔄 New Chat
        </button>
      </div>

      {/* Messages */}
      <div className="chatbot-messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>Ask about IC50 values, drug predictions, recommendations, or the preprocessing pipeline.</p>
          </div>
        )}
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="message-role">{msg.role === 'user' ? '👤 You' : '🤖 AI'}</div>
            <div className="message-content">
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="chatbot-input">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Ask about IC50, predictions, recommendations..."
          disabled={loading}
          maxLength={2000}
        />
        <button onClick={sendMessage} disabled={loading || !input.trim()}>
          {loading ? '⏳' : '📤'} Send
        </button>
      </div>
    </div>
  );
}

export default ChatBot;
```

---

## Chat with Prediction Context

When a user makes a prediction, you can pass the prediction data as context to get a **specific, tailored** AI explanation:

```jsx
function PredictionResultWithChat({ prediction }) {
  const [showChat, setShowChat] = useState(false);
  const [conversationId, setConversationId] = useState(null);

  const askAboutPrediction = async () => {
    setShowChat(true);

    const response = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: 'What does this prediction mean?',
        context: {
          predicted_ic50: prediction.predicted_ic50,
          model_used: prediction.model_used,
          cell_line_name: prediction.cell_line_name,
          drug_name: prediction.drug_name
        }
      })
    });

    const data = await response.json();
    setConversationId(data.conversation_id);
  };

  return (
    <div>
      {/* Prediction result display */}
      <div className="prediction-result">
        <h4>Prediction Result</h4>
        <p>IC50: {prediction.predicted_ic50}</p>
        <p>Model: {prediction.model_used}</p>
        <button onClick={askAboutPrediction}>💬 Explain This Prediction</button>
      </div>

      {/* Chat panel (shown when user clicks "Explain") */}
      {showChat && (
        <ChatBot
          initialConversationId={conversationId}
          initialContext={prediction}
        />
      )}
    </div>
  );
}
```

---

## Conversation Management

### ChatSession Class

A reusable class for managing chat state:

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

    // Store conversation_id for future messages
    if (data.conversation_id) {
      this.conversationId = data.conversation_id;
    }

    this.messages.push({ role: 'user', content: message });
    this.messages.push({ role: 'assistant', content: data.message });

    return data;
  }

  async getHistory() {
    if (!this.conversationId) return [];
    const response = await fetch(`${this.apiBase}/chat/history/${this.conversationId}`);
    const data = await response.json();
    return data.messages || [];
  }

  async clear() {
    if (!this.conversationId) return;
    await fetch(`${this.apiBase}/chat/history/${this.conversationId}`, { method: 'DELETE' });
    this.conversationId = null;
    this.messages = [];
  }
}
```

---

## State Management

### Using React Context

```jsx
// ChatContext.js
import { createContext, useContext, useState } from 'react';

const ChatContext = createContext();

export function ChatProvider({ children }) {
  const [session, setSession] = useState(new ChatSession());

  return (
    <ChatContext.Provider value={{ session, setSession }}>
      {children}
    </ChatContext.Provider>
  );
}

export function useChat() {
  return useContext(ChatContext);
}
```

### Using Zustand (Alternative)

```javascript
// store/chatStore.js
import { create } from 'zustand';

const useChatStore = create((set, get) => ({
  conversationId: null,
  messages: [],
  loading: false,

  sendMessage: async (message, context = null) => {
    set({ loading: true });
    // ... fetch logic
    set({ loading: false });
  },

  clearConversation: () => {
    set({ conversationId: null, messages: [] });
  }
}));
```

---

## Styling and UI

### Recommended CSS

```css
.chatbot-container {
  max-width: 800px;
  margin: 0 auto;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  overflow: hidden;
}

.chatbot-header {
  background: #1a73e8;
  color: white;
  padding: 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chatbot-messages {
  height: 400px;
  overflow-y: auto;
  padding: 16px;
}

.message.user {
  background: #e3f2fd;
  margin-bottom: 8px;
  padding: 12px;
  border-radius: 8px;
}

.message.assistant {
  background: #f5f5f5;
  margin-bottom: 8px;
  padding: 12px;
  border-radius: 8px;
}

.message-role {
  font-size: 0.8em;
  color: #666;
  margin-bottom: 4px;
}

.message-content {
  line-height: 1.4;
}

.chatbot-input {
  display: flex;
  padding: 16px;
  border-top: 1px solid #e0e0e0;
  gap: 8px;
}

.chatbot-input input {
  flex: 1;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.chatbot-input button {
  padding: 12px 24px;
  background: #1a73e8;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.chatbot-input button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.status-badge {
  font-size: 0.8em;
  padding: 4px 8px;
  border-radius: 4px;
}

.status-badge.available {
  background: #4caf50;
  color: white;
}

.status-badge.fallback {
  background: #ff9800;
  color: white;
}

.new-chat-btn {
  background: transparent;
  color: white;
  border: 1px solid white;
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
}
```

---

## Error Handling

```jsx
const sendMessage = async () => {
  try {
    const response = await fetch(`${API_BASE}/chat`, { ... });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      return {
        success: false,
        message: 'Unable to connect to the chatbot service. Please check your network connection.',
        model_used: 'error'
      };
    }
    return {
      success: false,
      message: error.message,
      model_used: 'error'
    };
  }
};
```

---

## Testing

```jsx
// Test basic message
const result1 = await session.send('What does IC50 mean?');
assert(result1.success === true);
assert(result1.conversation_id !== null);

// Test with context
const result2 = await session.send('Explain this prediction', {
  predicted_ic50: -1.46,
  model_used: 'CatBoost'
});
 });
assert(result2.success === true);

 assert(result2.message.includes('-1.46'));

// Test conversation continuity
const result3 = await session.send('More details?', null, result2.conversation_id);
assert(result3.success === true);
```

---

## Production Considerations

1. **CORS**: Restr `allow_origins` to your production domain
2. **Rate limiting**: Consider adding rate limiting middleware
3. **Authentication**: Add authentication if needed
4. **HTTPS**: Use HTTPS in production
5. **Error monitoring**: Set up error tracking (Sentry, etc.)
6. **Load testing**: Test with concurrent users
7. **Database persistence**: Consider replacing in-memory storage with a database for conversation history