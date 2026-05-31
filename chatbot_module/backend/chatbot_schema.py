"""
Chatbot request and response schemas for API validation.

These schemas define the structure for chatbot interactions:
- ChatRequest: User message input
- ChatResponse: AI assistant response
- ChatMessage: Individual message in conversation history
- ChatHistoryResponse: Full conversation history response
- ChatbotStatusResponse: Service status check response
- ErrorResponse: Standard error response format

This is the REAL schema from the production implementation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class ChatRequest(BaseModel):
    """Request schema for chatbot interaction."""
    
    message: str = Field(
        ..., 
        description="User message to the chatbot",
        min_length=1,
        max_length=2000
    )
    
    conversation_id: Optional[str] = Field(
        None, 
        description="Optional conversation ID for context continuity"
    )
    
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional context data (e.g., prediction results, cell line info)"
    )
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        """Ensure message is not empty after stripping whitespace."""
        if not v.strip():
            raise ValueError("Message cannot be empty or contain only whitespace")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What does an IC50 value of -1.46 mean?",
                "conversation_id": "conv_abc123",
                "context": {
                    "predicted_ic50": -1.46,
                    "model_used": "CatBoost",
                    "cell_line_name": "A172",
                    "drug_name": "Camptothecin"
                }
            }
        }


class ChatMessage(BaseModel):
    """Individual message in a conversation."""
    
    role: str = Field(
        ..., 
        description="Message role: 'user' or 'assistant'",
    )
    
    content: str = Field(
        ..., 
        description="Message content"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Message timestamp"
    )
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        """Ensure role is either 'user' or 'assistant'."""
        if v not in ('user', 'assistant'):
            raise ValueError("Role must be either 'user' or 'assistant'")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "assistant",
                "content": "An IC50 value of -1.46 indicates high drug sensitivity...",
                "timestamp": "2026-05-23T12:00:00Z"
            }
        }


class ChatResponse(BaseModel):
    """Response schema for chatbot interaction."""
    
    success: bool = Field(..., description="Whether the chatbot response was successful")
    message: str = Field(..., description="AI assistant response message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context continuity")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    model_used: Optional[str] = Field(None, description="Gemini model used for response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "An IC50 value of -1.46 indicates that the drug Camptothecin is highly effective for the A172 cell line. Lower IC50 values mean the drug requires a lower concentration to inhibit cell growth, which is a sign of strong drug sensitivity.",
                "conversation_id": "conv_abc123",
                "timestamp": "2026-05-23T12:00:00Z",
                "model_used": "gemini-2.0-flash"
            }
        }


class ChatHistoryResponse(BaseModel):
    """Response schema for chat history retrieval."""
    
    success: bool = Field(..., description="Whether history retrieval was successful")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    messages: List[ChatMessage] = Field(default_factory=list, description="List of conversation messages")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "conversation_id": "conv_abc123",
                "messages": [
                    {
                        "role": "user",
                        "content": "What does IC50 mean?",
                        "timestamp": "2026-05-23T11:55:00Z"
                    },
                    {
                        "role": "assistant",
                        "content": "IC50 (Half Maximal Inhibitory Concentration) is...",
                        "timestamp": "2026-05-23T11:55:05Z"
                    }
                ],
                "timestamp": "2026-05-23T12:00:00Z"
            }
        }


class ChatbotStatusResponse(BaseModel):
    """Response schema for chatbot status check."""
    
    available: bool = Field(..., description="Whether the chatbot service is available")
    model: Optional[str] = Field(None, description="Gemini model name if available")
    api_key_configured: bool = Field(..., description="Whether the Gemini API key is configured")
    message: str = Field(..., description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "available": True,
                "model": "gemini-2.0-flash",
                "api_key_configured": True,
                "message": "Chatbot service is available"
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    
    success: bool = Field(default=False, description="Always False for error responses")
    error: str = Field(..., description="Error type")
    detail: Optional[str] = Field(None, description="Error detail message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Chatbot error",
                "detail": "Gemini API call failed: quota exceeded"
            }
        }