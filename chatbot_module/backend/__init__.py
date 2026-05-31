"""
Chatbot Module — Backend Package

This package contains the complete chatbot backend implementation
for the Drug Response Prediction and Recommendation Platform.

Modules:
- chatbot_service: Core service with Gemini integration, fallback logic, and singleton pattern
- chatbot_router: FastAPI router with all chatbot endpoints
- chatbot_schema: Pydantic request/response models for API validation
- chatbot_memory: Conversation history management (in-memory storage)
- chatbot_utils: Environment loading, fallback responses, topic matching

Usage:
    from chatbot_module.backend.chatbot_router import router
    app.include_router(router)
"""

from chatbot_module.backend.chatbot_service import ChatbotService, get_chatbot_service
from chatbot_module.backend.chatbot_router import router
from chatbot_module.backend.chatbot_schema import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    ChatHistoryResponse,
    ChatbotStatusResponse,
    ErrorResponse
)
from chatbot_module.backend.chatbot_memory import ConversationMemory
from chatbot_module.backend.chatbot_utils import FALLBACK_RESPONSES, match_fallback_topic

__all__ = [
    "ChatbotService",
    "get_chatbot_service",
    "router",
    "ChatRequest",
    "ChatResponse",
    "ChatMessage",
    "ChatHistoryResponse",
    "ChatbotStatusResponse",
    "ErrorResponse",
    "ConversationMemory",
    "FALLBACK_RESPONSES",
    "match_fallback_topic",
]