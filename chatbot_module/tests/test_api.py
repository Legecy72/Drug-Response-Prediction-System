"""
Tests for the Chatbot API endpoints.

These tests validate:
1. POST /chat endpoint
2. GET /chat/status endpoint
3. GET /chat/history/{conversation_id} endpoint
4. DELETE /chat/history/{conversation_id} endpoint
5. Request validation (empty message, too long message)
6. Error handling
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chatbot_module.backend.chatbot_router import router
from chatbot_module.backend.chatbot_schema import ChatRequest, ChatResponse, ChatbotStatusResponse
from chatbot_module.backend.chatbot_service import ChatbotService, get_chatbot_service


# ============================================================================
# Test Client Setup
# ============================================================================

@pytest.fixture(scope="module")
def test_client():
    """Create a test client with the chatbot router."""
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    return TestClient(app)


# ============================================================================
# POST /chat Tests
# ============================================================================

class TestChatEndpoint:
    """Tests for the POST /chat endpoint."""

    @patch('chatbot_module.backend.chatbot_service.ChatbotService._initialize_gemini')
    def _mock_gemini_init(self):
        """Mock Gemini initialization."""
        self.api_key_configured = True
        self.gemini_model = MagicMock()

    def test_chat_basic_message(self, test_client):
        """Test sending a basic chat message."""
        response = test_client.post(
            "/chat",
            json={
                "message": "What does IC50 mean?"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "IC50" in data["message"]
        assert data["conversation_id"] is not None

    def test_chat_with_context(self, test_client):
        """Test sending a chat message with prediction context."""
        response = test_client.post(
            "/chat",
            json={
                "message": "What does this prediction mean?",
                "conversation_id": "conv_test_123",
                "context": {
                    "predicted_ic50": -1.46,
                    "model_used": "CatBoost",
                    "cell_line_name": "A172",
                    "drug_name": "Camptothecin"
                }
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["conversation_id"] == "conv_test_123"

    def test_chat_with_conversation_id(self, test_client):
        """Test chat with existing conversation ID."""
        # First message
        response1 = test_client.post(
            "/chat",
            json={
                "message": "Hello",
                "conversation_id": "conv_abc123"
            }
        )
        assert response1.status_code == 200

        # Second message (same conversation)
        response2 = test_client.post(
            "/chat",
            json={
                "message": "What is IC50?",
                "conversation_id": "conv_abc123"
            }
        )
        assert response2.status_code == 200
        assert response2.json()["conversation_id"] == "conv_abc123"

    def test_chat_empty_message_validation(self, test_client):
        """Test that empty messages are rejected."""
        response = test_client.post(
            "/chat",
            json={
                "message": "   "
            }
        )

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_chat_message_too_long_validation(self, test_client):
        """Test that messages exceeding 2000 characters are rejected."""
        response = test_client.post(
            "/chat",
            json={
                "message": "x" * 2001
            }
        )

        assert response.status_code == 422

    def test_chat_message_whitespace_stripped(self, test_client):
        """Test that message whitespace is stripped."""
        response = test_client.post(
            "/chat",
            json={
                "message": "  What is IC50?  "
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


# ============================================================================
# GET /chat/status Tests
# ============================================================================

class TestChatStatusEndpoint:
    """Tests for the GET /chat/status endpoint."""

    @patch('chatbot_module.backend.chatbot_service.ChatbotService._initialize_gemini')
    def _mock_gemini_init(self):
        """Mock Gemini initialization."""
        self.api_key_configured = True
        self.gemini_model = MagicMock()

    def test_chat_status_available(self, test_client):
        """Test chatbot status when Gemini is available."""
        response = test_client.get("/chat/status")

        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True
        assert data["model"] is not None
        assert data["api_key_configured"] is True

    def test_chat_status_fallback_mode(self, test_client):
        """Test chatbot status in fallback mode."""
        # Override the service to be in fallback mode
        chatbot_service = get_chatbot_service()
        chatbot_service.api_key_configured = False
        chatbot_service.gemini_model = None

        response = test_client.get("/chat/status")

        assert response.status_code == 200
        data = response.json()
        assert data["available"] is False
        assert data["model"] is None
        assert data["api_key_configured"] is False


# ============================================================================
# GET /chat/history/{conversation_id} Tests
# ============================================================================

class TestChatHistoryEndpoint:
    """Tests for the GET /chat/history/{conversation_id} endpoint."""

    @patch('chatbot_module.backend.chatbot_service.ChatbotService._initialize_gemini')
    def _mock_gemini_init(self):
        """Mock Gemini initialization."""
        self.api_key_configured = True
        self.gemini_model = MagicMock()

    def test_get_history_existing_conversation(self, test_client):
        """Test retrieving history for an existing conversation."""
        # First, create a conversation
        test_client.post(
            "/chat",
            json={
                "message": "Hello",
                "conversation_id": "conv_history_test"
            }
        )

        # Then retrieve history
        response = test_client.get("/chat/history/conv_history_test")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["conversation_id"] == "conv_history_test"
        assert len(data["messages"]) >= 1

    def test_get_history_unknown_conversation(self, test_client):
        """Test retrieving history for a non-existent conversation."""
        response = test_client.get("/chat/history/nonexistent_conv_id")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["conversation_id"] == "nonexistent_conv_id"
        assert data["messages"] == []


# ============================================================================
# DELETE /chat/history/{conversation_id} Tests
# ============================================================================

class TestChatHistoryClearEndpoint:
    """Tests for the DELETE /chat/history/{conversation_id} endpoint."""

    @patch('chatbot_module.backend.chatbot_service.ChatbotService._initialize_gemini')
    def _mock_gemini_init(self):
        """Mock Gemini initialization."""
        self.api_key_configured = True
        self.gemini_model = MagicMock()

    def test_clear_history_existing_conversation(self, test_client):
        """Test clearing history for an existing conversation."""
        # First, create a conversation
        test_client.post(
            "/chat",
            json={
                "message": "Hello",
                "conversation_id": "conv_clear_test"
            }
        )

        # Then clear it
        response = test_client.delete("/chat/history/conv_clear_test")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "cleared" in data["message"]

    def test_clear_history_nonexistent_conversation(self, test_client):
        """Test clearing history for a non-existent conversation."""
        response = test_client.delete("/chat/history/nonexistent_conv_id")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "not found" in data["message"]