"""
Chatbot router for FastAPI.

This router handles:
- POST /chat — Chat interactions with the AI assistant
- GET /chat/history/{conversation_id} — Conversation history retrieval
- DELETE /chat/history/{conversation_id} — Clear conversation history
- GET /chat/status — Chatbot status checks

This is the REAL router from the production implementation,
organized for standalone use within the chatbot_module package.
"""

from fastapi import APIRouter, HTTPException, status
from typing import Optional

from chatbot_module.backend.chatbot_schema import (
    ChatRequest,
    ChatResponse,
    ChatHistoryResponse,
    ChatbotStatusResponse,
    ErrorResponse
)
from chatbot_module.backend.chatbot_service import get_chatbot_service

router = APIRouter(
    prefix="/chat",
    tags=["chatbot"],
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)

chatbot_service = get_chatbot_service()


@router.post(
    "",
    response_model=ChatResponse,
    summary="Send a message to the AI chatbot",
    description="Send a message to the AI assistant and receive a domain-aware response about drug response prediction, IC50 values, recommendations, and more."
)
async def chat_with_assistant(request: ChatRequest):
    """
    Chat with the AI assistant.
    
    The chatbot can explain:
    - IC50 values and their interpretation
    - Drug sensitivity and resistance
    - Recommendation scores and ranking
    - Preprocessing pipeline (encoding, feature selection, scaling)
    - ML models (CatBoost, XGBoost, LightGBM, etc.)
    - Project architecture
    
    **Safety**: This chatbot does NOT provide medical diagnosis or treatment recommendations.
    It is for educational and explainability purposes only.
    
    - **message**: Your question or message (required, max 2000 characters)
    - **conversation_id**: Optional ID to continue a previous conversation
    - **context**: Optional context data (e.g., prediction results)
    
    Returns AI assistant response with conversation ID for continuity.
    """
    try:
        result = chatbot_service.chat(
            message=request.message,
            conversation_id=request.conversation_id,
            context=request.context
        )
        
        return ChatResponse(
            success=result["success"],
            message=result["message"],
            conversation_id=result.get("conversation_id"),
            model_used=result.get("model_used")
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chatbot error: {str(e)}"
        )


@router.get(
    "/history/{conversation_id}",
    response_model=ChatHistoryResponse,
    summary="Get conversation history",
    description="Retrieve the message history for a specific conversation."
)
async def get_conversation_history(conversation_id: str):
    """
    Get conversation history for a given conversation ID.
    
    - **conversation_id**: The conversation ID to retrieve history for
    
    Returns list of messages in the conversation.
    """
    try:
        result = chatbot_service.get_conversation_history(conversation_id)
        
        return ChatHistoryResponse(
            success=result["success"],
            conversation_id=result.get("conversation_id"),
            messages=result.get("messages", [])
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversation history: {str(e)}"
        )


@router.delete(
    "/history/{conversation_id}",
    summary="Clear conversation history",
    description="Clear the message history for a specific conversation."
)
async def clear_conversation_history(conversation_id: str):
    """
    Clear conversation history for a given conversation ID.
    
    - **conversation_id**: The conversation ID to clear
    
    Returns confirmation that the conversation was cleared.
    """
    try:
        result = chatbot_service.clear_conversation(conversation_id)
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear conversation history: {str(e)}"
        )


@router.get(
    "/status",
    response_model=ChatbotStatusResponse,
    summary="Check chatbot status",
    description="Check if the chatbot service is available and configured."
)
async def chatbot_status():
    """
    Check chatbot service status.
    
    Returns whether the Gemini API is configured and available,
    along with the model name and configuration status.
    """
    try:
        status_info = chatbot_service.get_status()
        
        return ChatbotStatusResponse(
            available=status_info["available"],
            model=status_info.get("model"),
            api_key_configured=status_info["api_key_configured"],
            message=status_info["message"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check chatbot status: {str(e)}"
        )