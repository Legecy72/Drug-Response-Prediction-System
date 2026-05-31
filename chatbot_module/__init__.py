"""
Chatbot Module — Package Init

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

__version__ = "1.0.0"
__author__ = "Drug Response Prediction Team"