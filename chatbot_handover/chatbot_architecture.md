# Chatbot Subsystem Architecture

## Architecture Explanation

The chatbot subsystem follows a **layered Architecture** pattern with five distinct layers:

### Layer 1 — Presentation Layer (Frontend)
The **Frontend Chat Widget UI** is the user-facing component embedded in the Drug Response Prediction web Application. It provides a chat interface where users type natural-language questions about drug response predictions, IC50 values, recommendation scores, and project methodology. The widget sends HTTP requests to the FastAPI backend and renders AI-generated responses in a conversational UI format.

### Layer 2 — API Gateway Layer (FastAPI Router)
The **Chatbot Router** (`api/routers/chatbot.py`) is a FastAPI `APIRouter` that exposes four REST endpoints:
- `POST /chat` — Accepts a `ChatRequest` (message, optional conversation_id, optional context) and returns a `ChatResponse`
- `GET /chat/history/{conversation_id}` — Retrieves conversation history
- `DELETE /chat/history/{conversation_id}` — Clears conversation history
- `GET /chat/status` — Returns service availability status

The router validates request schemas using Pydantic (`ChatRequest`, `ChatResponse`, `ChatHistoryResponse`, `ChatbotStatusResponse`) and delegates business logic to the **Chatbot Service**.

### Layer 3 — Business Logic Layer (Chatbot Service)
The **Chatbot Service** (`api/services/chatbot_service.py`) is the core orchestrator implementing a **Singleton Pattern** via `get_chatbot_service()`. It manages:
1. **Gemini API Integration** — Initializes a `google.generativeai.GenerativeModel` with the domain-specific system prompt, sends enhanced user messages (with injected context), and receives AI-generated responses
2. **Model Fallback Chain** — On quota errors (HTTP 429), automatically tries fallback models: `gemini-2.5-flash` → `gemini-2.0-flash-lite` → `gemini-2.0-flash`
3. **Conversation History Management** — Stores messages in an in-memory `Dict[str, List[Dict]]` keyed by `conversation_id`, limited to 50 messages per conversation, with automatic trimming of older messages
4. **Context Injection** — Prepends prediction/recommendation results to user messages before sending to Gemini, enabling domain-aware responses
5. **Fallback Response Engine** — When Gemini API is unavailable (missing API key, import failure, or all models exhausted), selects pre-defined responses by topic-matching user message keywords

### Layer 4 — AI Model Layer (Gemini API)
The **Gemini API Integration** connects to Google's Generative AI service using the `google-generativeai` Python SDK. The integration:
- Configures the API with the `GEMINI_API_KEY` from the `.env` file
- Creates a `GenerativeModel` instance with `system_instruction` set to the domain-specific system prompt
- Uses `start_chat()` with conversation history formatted for Gemini's `user/model` role convention
- Sends enhanced messages via `send_message()` and extracts text from response candidates

### Layer 5 — Data & Configuration Layer
- **System Prompt** (`chatbot_module/prompts/system_prompt.txt`) — Defines the AI assistant's persona, capabilities, project knowledge, safety restrictions, and response format. Loaded at runtime; falls back to an embedded constant if the file is missing
- **Conversation Memory** — In-memory dictionary storing conversation histories keyed by UUID conversation IDs, with a 50-message cap per conversation
- **Fallback Responses** (`FALLBACK_RESPONSES` dict) — Pre-defined topic-specific responses for IC50, recommendations, preprocessing, models, and general queries
- **Environment Configuration** (`.env`) — Stores `GEMINI_API_KEY` and `GEMINI_MODEL` settings

---

## Mermaid Architecture Diagram

```mermaid
graph TB
    subgraph PRESENTATION["Presentation Layer — Frontend"]
        direction LR
        User["👤 User"]
        ChatUI["💬 Chat Widget UI<br/>web/js/app.js"]
    end

    subgraph API_GATEWAY["API Gateway Layer — FastAPI"]
        direction LR
        Router["🔀 Chatbot Router<br/>api/routers/chatbot.py<br/>POST /chat<br/>GET /chat/history<br/>DELETE /chat/history<br/>GET /chat/status"]
        Schemas["📋 Pydantic Schemas<br/>api/schemas/chatbot.py<br/>ChatRequest<br/>ChatResponse<br/>ChatHistoryResponse<br/>ChatbotStatusResponse"]
    end

    subgraph BUSINESS_LOGIC["Business Logic Layer — Service"]
        direction LR
        Service["🧠 Chatbot Service<br/>api/services/chatbot_service.py<br/>Singleton Pattern"]
        PromptLoader["📝 System Prompt Loader<br/>chatbot_module/prompts/system_prompt.txt"]
        ContextInjector["💉 Context Injector<br/>Prepends prediction/recommendation<br/>results to user message"]
        FallbackEngine["🛡️ Fallback Response Engine<br/>FALLBACK_RESPONSES dict<br/>_match_fallback_topic()"]
        ModelFallback["🔄 Model Fallback Chain<br/>gemini-2.5-flash →<br/>gemini-2.0-flash-lite →<br/>gemini-2.0-flash"]
    end

    subgraph AI_MODEL["AI Model Layer — Gemini"]
        direction LR
        GeminiAPI["🔮 Gemini API Integration<br/>google-generativeai SDK"]
        GeminiModel["⚡ Gemini Model<br/>GenerativeModel with<br/>system_instruction"]
    end

    subgraph DATA_CONFIG["Data & Configuration Layer"]
        direction LR
        EnvConfig["⚙️ Environment Config<br/>.env file<br/>GEMINI_API_KEY<br/>GEMINI_MODEL"]
        ConvMemory["💾 Conversation Memory<br/>In-memory Dict<br/>50 msg/conversation cap"]
        FallbackData["📦 Fallback Responses<br/>ic50, recommendation,<br/>preprocessing, models, general"]
        SystemPrompt["📄 System Prompt<br/>Domain persona, capabilities,<br/>safety rules, response format"]
    end

    %% Communication Paths
    User -->|"HTTP Request<br/>POST /chat"| ChatUI
    ChatUI -->|"HTTP Request<br/>POST /chat<br/>with ChatRequest body"| Router
    Router -->|"Validate & Route"| Schemas
    Schemas -->|"Validated Request"| Service
    Service -->|"Load System Prompt"| PromptLoader
    PromptLoader -->|"System Prompt String"| Service
    Service -->|"Inject Context<br/>if context dict provided"| ContextInjector
    ContextInjector -->|"Enhanced Message<br/>[Context: ...]<br/>User question: ..."| Service
    Service -->|"Check Availability<br/>is_available()"| GeminiAPI
    GeminiAPI -->|"Available: True"| Service
    Service -->|"Send Enhanced Message<br/>with conversation history"| GeminiModel
    GeminiModel -->|"AI Response Text"| Service

    %% Fallback Path
    Service -->|"Available: false<br/>OR API call fails"| FallbackEngine
    FallbackEngine -->|"Match Topic<br/>_match_fallback_topic()"| FallbackData
    FallbackData -->|"Pre-defined Response"| FallbackEngine
    FallbackEngine -->|"Fallback Response"| Service

    %% Model Fallback Path
    Service -->|"Quota Error (429)<br/>on primary model"| ModelFallback
    ModelFallback -->|"Try Next Model<br/>in fallback chain"| GeminiAPI
    GeminiAPI -->|"New Model Instance"| GeminiModel

    %% Storage Paths
    Service -->|"Store User Message<br/>add_message(role=user)"| ConvMemory
    Service -->|"Store AI Response<br/>add_message(role=assistant)"| ConvMemory
    Service -->|"Retrieve History<br/>build_gemini_history()"| ConvMemory

    %% Configuration Paths
    EnvConfig -->|"GEMINI_API_KEY<br/>GEMINI_MODEL"| GeminiAPI
    EnvConfig -->|"GEMINI_API_KEY<br/>GEMINI_MODEL"| Service

    %% Response Path
    Service -->|"ChatResponse Dict"| Router
    Router -->|"HTTP Response<br/>ChatResponse JSON"| ChatUI
    ChatUI -->|"Display AI Response"| User

    %% Styling
    classDef user fill:#E8F5E9,stroke:#333,stroke-width:2px,color:#333
    classDef frontend fill:#4FC3F7,stroke:#333,stroke-width:2px,color:#333
    classDef router fill:#FFD700,stroke:#333,stroke-width:2px,color:#333
    classDef schema fill:#87CEEB,stroke:#333,stroke-width:2px,color:#333
    classDef service fill:#FF6B6B,stroke:#333,stroke-width:2px,color:#333
    classDef prompt fill:#A8E6CF,stroke:#333,stroke-width:2px,color:#333
    classDef context fill:#DDA0DD,stroke:#333,stroke-width:2px,color:#333
    classDef fallback fill:#FFA500,stroke:#333,stroke-width:2px,color:#333
    classDef modelFallback fill:#FF4500,stroke:#333,stroke-width:2px,color:#333
    classDef gemini fill:#9370DB,stroke:#333,stroke-width:2px,color:#333
    classDef geminiModel fill:#6A0DAD,stroke:#333,stroke-width:2px,color:#333
    classDef env fill:#98FB98,stroke:#333,stroke-width:2px,color:#333
    classDef memory fill:#B0C4DE,stroke:#333,stroke-width:2px,color:#333
    classDef fallbackData fill:#F0E68C,stroke:#333,stroke-width:2px,color:#333
    classDef systemPrompt fill:#C1FFC1,stroke:#333,stroke-width:2px,color:#333

    class User user
    class ChatUI frontend
    class Router router
    class Schemas schema
    class Service service
    class PromptLoader prompt
    class ContextInjector context
    class FallbackEngine fallback
    class ModelFallback modelFallback
    class GeminiAPI gemini
    class GeminiModel geminiModel
    class EnvConfig env
    class ConvMemory memory
    class FallbackData fallbackData
    class SystemPrompt systemPrompt