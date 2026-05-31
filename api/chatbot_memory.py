"""
Conversation memory management for the Drug Response Prediction chatbot.

This module provides:
- In-memory conversation history storage
- History size limiting (max 50 messages per conversation)
- Conversation retrieval and clearing

Extracted from the original ChatbotService class for clean modularity.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Maximum number of messages to keep per conversation
MAX_HISTORY_SIZE = 50


class ConversationMemory:
    """
    Manages conversation history for the chatbot service.
    
    Stores messages in-memory keyed by conversation_id.
    Each message has a role ('user' or 'assistant'), content, and timestamp.
    
    Attributes:
        conversations: Dictionary of conversation histories keyed by conversation_id
    """
    
    def __init__(self, max_history: int = MAX_HISTORY_SIZE):
        """
        Initialize the conversation memory.
        
        Args:
            max_history: Maximum number of messages to keep per conversation.
                         Older messages are trimmed when the limit is exceeded.
        """
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        self.max_history = max_history
    
    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """
        Add a message to conversation history.
        
        Args:
            conversation_id: Conversation ID
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        self.conversations[conversation_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit history size to prevent memory issues
        if len(self.conversations[conversation_id]) > self.max_history:
            self.conversations[conversation_id] = (
                self.conversations[conversation_id][-self.max_history:]
            )
    
    def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """
        Get the raw conversation history list for a given conversation ID.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            List of message dictionaries with role, content, and timestamp
        """
        return self.conversations.get(conversation_id, [])
    
    def get_formatted_history(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get conversation history formatted for API response.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Dictionary with conversation_id and messages list
        """
        history = self.get_history(conversation_id)
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "messages": [
                {
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg["timestamp"]
                }
                for msg in history
            ]
        }
    
    def build_gemini_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Build conversation history formatted for the Gemini API.
        
        The Gemini API requires roles 'user' and 'model' (not 'assistant').
        This method converts 'assistant' roles to 'model' and formats
        messages with the 'parts' structure that Gemini expects.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            List of message dictionaries in Gemini API format:
            [{"role": "user"|"model", "parts": [content]}, ...]
        """
        history = self.get_history(conversation_id)
        
        return [
            {"role": "model" if h["role"] == "assistant" else h["role"], "parts": [h["content"]]}
            for h in history
        ] if history else []
    
    def clear_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Clear conversation history for a given conversation ID.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Dictionary confirming the conversation was cleared
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return {
                "success": True,
                "message": f"Conversation {conversation_id} cleared"
            }
        return {
            "success": True,
            "message": f"Conversation {conversation_id} not found (no action needed)"
        }
    
    def get_active_conversation_count(self) -> int:
        """
        Get the number of active conversations.
        
        Returns:
            Number of conversations currently stored
        """
        return len(self.conversations)
