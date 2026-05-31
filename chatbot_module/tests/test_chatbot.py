"""
Tests for the ChatbotService — Core service logic.

These tests validate:
1. Chat message processing (with and without context)
2. Conversation history management
3. Fallback response selection
4. Gemini API integration (mocked)
5. Model fallback on quota errors
6. Singleton pattern
7. Status reporting
"""

import os
import pytest
import uuid
from unittest.mock import patch, MagicMock

from datetime import datetime

# Import the real service components
from chatbot_module.backend.chatbot_service import ChatbotService, get_chatbot_service
from chatbot_module.backend.chatbot_memory import ConversationMemory
from chatbot_module.backend.chatbot_utils import FALLBACK_RESPONSES, match_fallback_topic
from chatbot_module.backend.chatbot_schema import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    ChatHistoryResponse,
    ChatbotStatusResponse
)


# ============================================================================
# ConversationMemory Tests
# ============================================================================

class TestConversationMemory:
    """Tests for the ConversationMemory class."""

    def test_add_message(self):
        """Test adding messages to conversation history."""
        memory = ConversationMemory()

        # Add user message
        memory.add_message("conv_123", "user", "What does IC50 mean?")

        # Add assistant message
        memory.add_message("conv_123", "assistant", "IC50 measures drug effectiveness...")

        # Verify history
        history = memory.get_history("conv_123")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "What does IC50 mean?"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "IC50 measures drug effectiveness..."

    def test_max_history_limit(self):
        """Test that history is trimmed to max 50 messages."""
        memory = ConversationMemory(max_history=50)

        # Add 60 messages
        for i in range(60):
            memory.add_message("conv_123", "user", f"Message {i}")
            memory.add_message("conv_123", "assistant", f"Response {i}")

        # Verify only 50 messages are kept (last 50)
        history = memory.get_history("conv_123")
        assert len(history) == 50

    def test_get_formatted_history(self):
        """Test formatted history retrieval."""
        memory = ConversationMemory()

        memory.add_message("conv_123", "user", "Hello")
        memory.add_message("conv_123", "assistant", "Hi there")

        formatted = memory.get_formatted_history("conv_123")
        assert formatted["success"] is True
        assert formatted["conversation_id"] == "conv_123"
        assert len(formatted["messages"]) == 2
        assert formatted["messages"][0]["role"] == "user"
        assert formatted["messages"][1]["role"] == "assistant"

    def test_build_gemini_history(self):
        """Test Gemini API history format conversion.

        The Gemini API requires 'model' role (not 'assistant') and 'parts' structure.
        """
        memory = ConversationMemory()

        memory.add_message("conv_123", "user", "What is IC50?")
        memory.add_message("conv_123", "assistant", "IC50 is a measure...")

        gemini_history = memory.build_gemini_history("conv_123")
        assert len(gemini_history) == 2
        assert gemini_history[0]["role"] == "user"
        assert gemini_history[0]["parts"] == ["What is IC50?"]
        assert gemini_history[1]["role"] == "model"  # 'assistant' -> 'model'
        assert gemini_history[1]["parts"] == ["IC50 is a measure..."]

    def test_clear_conversation(self):
        """Test clearing conversation history."""
        memory = ConversationMemory()

        memory.add_message("conv_123", "user", "Hello")

        # Clear existing conversation
        result = memory.clear_conversation("conv_123")
        assert result["success"] is True
        assert "conv_123 cleared" in result["message"]

        # Verify history is empty
        history = memory.get_history("conv_123")
        assert len(history) == 0

    def test_clear_nonexistent_conversation(self):
        """Test clearing a conversation that doesn't exist."""
        memory = ConversationMemory()

        result = memory.clear_conversation("nonexistent_conv")
        assert result["success"] is True
        assert "not found" in result["message"]

    def test_get_active_conversation_count(self):
        """Test counting active conversations."""
        memory = ConversationMemory()

        # No conversations yet
        assert memory.get_active_conversation_count() == 0

        # Add conversations
        memory.add_message("conv_1", "user", "Hello 1")
        memory.add_message("conv_2", "user", "Hello 2")
        memory.add_message("conv_3", "user", "Hello 3")

        assert memory.get_active_conversation_count() == 3

    def test_empty_history_returns_empty_list(self):
        """Test that unknown conversation_id returns empty history."""
        memory = ConversationMemory()

        history = memory.get_history("unknown_conv_id")
        assert history == []

        formatted = memory.get_formatted_history("unknown_conv_id")
        assert formatted["messages"] == []


# ============================================================================
# Fallback Response Tests
# ============================================================================

class TestFallbackResponses:
    """Tests for the fallback response system."""

    def test_match_ic50_topic(self):
        """Test matching IC50-related keywords."""
        assert match_fallback_topic("What does IC50 mean?") == "ic50"
        assert match_fallback_topic("Tell me about IC 50") == "ic50"
        assert match_fallback_topic("Explain half maximal inhibitory concentration") == "ic50"
        assert match_fallback_topic("What is sensitivity?") == "ic50"
        assert match_fallback_topic("LN_IC50 interpretation") == "ic50"

    def test_match_recommendation_topic(self):
        """Test matching recommendation-related keywords."""
        assert match_fallback_topic("How are drugs recommended?") == "recommendation"
        assert match_fallback_topic("Explain the ranking system") == "recommendation"
        assert match_fallback_topic("What is the score formula?") == "recommendation"
        assert match_fallback_topic("Which drug is the best?") == "recommendation"
        assert match_fallback_topic("Top drug recommendation") == "recommendation"

    def test_match_preprocessing_topic(self):
        """Test matching preprocessing-related keywords."""
        assert match_fallback_topic("What preprocessing is done?") == "preprocessing"
        assert match_fallback_topic("Explain label encoding") == "preprocessing"
        assert match_fallback_topic("How does feature selection work?") == "preprocessing"
        assert match_fallback_topic("Describe the scaling pipeline") == "preprocessing"

    def test_match_models_topic(self):
        """Test matching model-related keywords."""
        assert match_fallback_topic("What models are available?") == "models"
        assert match_fallback_topic("Tell me about CatBoost") == "models"
        assert match_fallback_topic("Explain XGBoost") == "models"
        assert match_fallback_topic("How does LightGBM work?") == "models"
        assert match_fallback_topic("Compare Ridge and Lasso") == "models"

    def test_match_general_topic(self):
        """Test that unmatched queries return 'general'."""
        assert match_fallback_topic("Hello!") == "general"
        assert match_fallback_topic("What can you do?") == "general"
        assert match_fallback_topic("Tell me about the project") == "general"
        assert match_fallback_topic("How does this work?") == "general"

    def test_fallback_response_content(self):
        """Test that fallback responses contain expected content."""
        assert "IC50" in FALLBACK_RESPONSES["ic50"]
        assert "concentration" in FALLBACK_RESPONSES["ic50"]
        assert "recommendation" in FALLBACK_RESPONSES["recommendation"]
        assert "score" in FALLBACK_RESPONSES["recommendation"]
        assert "preprocessing" in FALLBACK_RESPONSES["preprocessing"]
        assert "Label Encoding" in FALLBACK_RESPONSES["preprocessing"]
        assert "CatBoost" in FALLBACK_RESPONSES["models"]
        assert "regression" in FALLBACK_RESPONSES["models"]
        assert "educational" in FALLBACK_RESPONSES["general"]
        assert "GEMINI_API_KEY" in FALLBACK_RESPONSES["general"]


# ============================================================================
# ChatbotService Tests
# ============================================================================

class TestChatbotService:
    """Tests for the ChatbotService class."""

    @patch('chatbot_module.backend.chatbot_service.ChatbotService._initialize_gemini')
    def _mock_gemini_init(self):
        """Mock Gemini initialization to avoid real API calls."""
        self.api_key_configured = True
        self.gemini_model = MagicMock()

    def test_chat_basic_message(self):
        """Test basic chat message without context."""
        service = ChatbotService()

        result = service.chat("What does IC50 mean?")

        assert result["success"] is True
        assert "IC50" in result["message"]
        assert result["conversation_id"] is not None
        assert result["model_used"] in ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash", "fallback"]

    def test_chat_with_context(self):
        """Test chat message with prediction context."""
        service = ChatbotService()

        context = {
            "predicted_ic50": -1.46,
            "model_used": "CatBoost",
            "cell_line_name": "A172",
            "drug_name": "Camptothecin"
        }

        result = service.chat("What does this prediction mean?", context=context)

        assert result["success"] is True
        assert result["conversation_id"] is not None
        # The enhanced message should include context
        assert "-1.46" in result["message"] or "IC50" in result["message"]

    def test_chat_with_conversation_id(self):
        """Test chat with existing conversation ID for continuity."""
        service = ChatbotService()

        # First message
        result1 = service.chat("What does IC50 mean?", conversation_id="conv_abc123")
        assert result1["conversation_id"] == "conv_abc123"

        # Second message (same conversation)
        result2 = service.chat("Can you explain more?", conversation_id="conv_abc123")
        assert result2["conversation_id"] == "conv_abc123"

        # Verify history was stored
        history = service.memory.get_history("conv_abc123")
        assert len(history) >= 2  # At least user + assistant messages

    def test_chat_new_conversation_id_generation(self):
        """Test that new conversation IDs are generated when not provided."""
        service = ChatbotService()

        result = service.chat("Hello!")

        assert result["conversation_id"] is not None
        # Should be a UUID format
        try:
            uuid.UUID(result["conversation_id"])
        except ValueError:
            pytest.fail("conversation_id should be a valid UUID")

    def test_chat_fallback_mode(self):
        """Test chat in fallback mode (Gemini unavailable)."""
        # Create service without Gemini
        service = ChatbotService()
        service.api_key_configured = False
        service.gemini_model = None

        result = service.chat("What does IC50 mean?")

        assert result["success"] is True
        assert result["model_used"] == "fallback"
        assert "IC50" in result["message"]
        assert "Gemini AI service is currently unavailable" in result["message"]

    def test_chat_model_fallback_on_quota_error(self):
        """Test that the service falls back to alternative models on quota errors."""
        service = ChatbotService()

        call_count = 0

        def mock_call_quota_then_success(message, conv_id, model_name):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: quota error
                raise Exception("429 RESOURCE_EXHAUSTED quota exceeded")
            else:
                # Second call: success
                return "Successful response from fallback model"

        service._call_gemini_with_model = mock_call_quota_then_success

        result = service.chat("Test message")

        assert result["success"] is True
        assert call_count == 2  # Tried primary, then fallback
        assert "fallback" in result["model_used"] or "gemini" in result["model_used"]

    def test_get_status_available(self):
        """Test status check when Gemini is available."""
        service = ChatbotService()
        service.api_key_configured = True

        status = service.get_status()

        assert status["available"] is True
        assert status["model"] is not None
        assert status["api_key_configured"] is True
        assert "available" in status["message"]

    def test_get_status_fallback_mode(self):
        """Test status check when Gemini is unavailable."""
        service = ChatbotService()
        service.api_key_configured = False
        service.gemini_model = None

        status = service.get_status()

        assert status["available"] is False
        assert status["model"] is None
        assert status["api_key_configured"] is False
        assert "fallback" in status["message"]

    def test_get_conversation_history(self):
        """Test retrieving conversation history."""
        service = ChatbotService()

        # Chat a few messages
        service.chat("Hello", conversation_id="conv_test")
        service.chat("What is IC50?", conversation_id="conv_test")

        history = service.get_conversation_history("conv_test")

        assert history["success"] is True
        assert history["conversation_id"] == "conv_test"
        assert len(history["messages"]) >= 2

    def test_clear_conversation(self):
        """Test clearing conversation history."""
        service = ChatbotService()

        service.chat("Hello", conversation_id="conv_clear_test")

        result = service.clear_conversation("conv_clear_test")

        assert result["success"] is True
        assert "cleared" in result["message"]

    def test_singleton_pattern(self):
        """Test that get_chatbot_service returns the same instance."""
        service1 = get_chatbot_service()
        service2 = get_chatbot_service()

        assert service1 is service2


# ============================================================================
# Schema Validation Tests
# ============================================================================

class TestSchemaValidation:
    """Tests for Pydantic schema validation."""

    def test_chat_request_valid(self):
        """Test valid ChatRequest schema."""
        request = ChatRequest(
            message="What does IC50 mean?",
            conversation_id="conv_123",
            context={"predicted_ic50": -1.46}
        )
        assert request.message == "What does IC50 mean?"
        assert request.conversation_id == "conv_123"
        assert request.context["predicted_ic50"] == -1.46

    def test_chat_request_message_validation(self):
        """Test ChatRequest message validation."""
        # Empty message
        with pytest.raises(ValueError):
            ChatRequest(message="   ", conversation_id=None, context=None)

        # Too long message
        with pytest.raises(ValueError):
            ChatRequest(message="x" * 2001, conversation_id=None, context=None)

    def test_chat_request_message_strip(self):
        """Test that message whitespace is stripped."""
        request = ChatRequest(
            message="  What does IC50 mean?  ",
            conversation_id=None,
            context=None
        )
        assert request.message == "What does IC50 mean?"

    def test_chat_message_role_validation(self):
        """Test ChatMessage role validation."""
        # Valid roles
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"

        msg = ChatMessage(role="assistant", content="Hi there")
        assert msg.role == "assistant"

        # Invalid role
        with pytest.raises(ValueError):
            ChatMessage(role="system", content="System message")

    def test_chat_response_schema(self):
        """Test ChatResponse schema."""
        response = ChatResponse(
            success=True,
            message="IC50 measures drug effectiveness",
            conversation_id="conv_123",
            model_used="gemini-2.5-flash"
        )
        assert response.success is True
        assert "IC50" in response.message
        assert response.conversation_id == "conv_123"

    def test_chatbot_status_response_schema(self):
        """Test ChatbotStatusResponse schema."""
        response = ChatbotStatusResponse(
            available=True,
            model="gemini-2.5-flash",
            api_key_configured=True,
            message="Chatbot service is available"
        )
        assert response.available is True
        assert response.model == "gemini-2.5-flash"