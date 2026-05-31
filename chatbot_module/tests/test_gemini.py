"""
Tests for Gemini API integration.

These tests validate:
1. Gemini API initialization
2. Model creation with system instruction
3. Chat session creation with history
4. Response text extraction
5. Empty response handling (safety filters)
6. Quota error handling (429)
7. Model fallback chain
8. Role mapping (assistant -> model)
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List, Optional

from chatbot_module.backend.chatbot_service import ChatbotService


# ============================================================================
# Gemini API Initialization Tests
# ============================================================================

class TestGeminiInitialization:
    """Tests for Gemini API initialization and configuration."""

    @patch('chatbot_module.backend.chatbot_service.ChatbotService._initialize_gemini')
    def _mock_gemini_init(self):
        """Mock Gemini initialization."""
        self.api_key_configured = True
        self.gemini_model = MagicMock()

    def test_gemini_initialization_with_api_key(self):
        """Test that Gemini initializes when API key is provided."""
        service = ChatbotService()

        # Mock the environment to have an API key
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key_123'}):
            service._initialize_gemini()

            assert service.api_key_configured is True
            assert service.gemini_model is not None

    def test_gemini_initialization_without_api_key(self):
        """Test that Gemini falls back when API key is missing."""
        service = ChatbotService()

        # Mock the environment without API key
        with patch.dict(os.environ, {'GEMINI_API_KEY': ''}):
            service._initialize_gemini()

            assert service.api_key_configured is False
            assert service.gemini_model is None

    def test_gemini_initialization_with_placeholder_key(self):
        """Test that Gemini falls back when API key is placeholder."""
        service = ChatbotService()

        with patch.dict(os.environ, {'GEMINI_API_KEY': 'your_gemini_api_key_here'}):
            service._initialize_gemini()

            assert service.api_key_configured is False

    def test_gemini_initialization_import_error(self):
        """Test that Gemini falls back when google-generativeai is not installed."""
        service = ChatbotService()

        # Mock ImportError for google.generativeai
        with patch('builtins.__import__', side_effect=ImportError("No module named 'google.generativeai'")):
            service._initialize_gemini()

            assert service.api_key_configured is False

    def test_gemini_model_creation_with_system_instruction(self):
        """Test that Gemini model is created with system instruction."""
        service = ChatbotService()

        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('google.generativeai.GenerativeModel') as mock_model_class:
                mock_model = MagicMock()
                mock_model_class.return_value = mock_model

                service._initialize_gemini()

                # Verify model was created with system instruction
                mock_model_class.assert_called_once()
                call_kwargs = mock_model_class.call_args[1]
                assert 'system_instruction' in call_kwargs
                assert call_kwargs['system_instruction'] is not None

    def test_gemini_model_name_configuration(self):
        """Test that Gemini model name is configurable via environment."""
        service = ChatbotService()

        with patch.dict(os.environ, {
            'GEMINI_API_KEY': 'test_key',
            'GEMINI_MODEL': 'gemini-2.0-flash'
        }):
            with patch('google.generativeai.GenerativeModel') as mock_model_class:
                mock_model = MagicMock()
                mock_model_class.return_value = mock_model

                service._initialize_gemini()

                # Verify model name was passed correctly
                call_kwargs = mock_model_class.call_args[1]
                assert call_kwargs['model_name'] == 'gemini-2.0-flash'


# ============================================================================
# Gemini API Call Tests
# ============================================================================

class TestGeminiAPICalls:
    """Tests for actual Gemini API call logic."""

    @patch('chatbot_module.backend.chatbot_service.ChatbotService._initialize_gemini')
    def _mock_gemini_init(self):
        """Mock Gemini initialization."""
        self.api_key_configured = True
        self.gemini_model = MagicMock()
        self._system_prompt = "Test system prompt"

    def test_call_gemini_basic(self):
        """Test basic Gemini API call."""
        service = ChatbotService()

        # Mock the Gemini response
        mock_response = MagicMock()
        mock_response.text = "Test response from Gemini"
        mock_response.candidates = [MagicMock()]

        mock_chat = MagicMock()
        mock_chat.send_message.return_value = mock_response

        service.gemini_model.start_chat.return_value = mock_chat

        result = service._call_gemini("Test message", "conv_123")

        assert result == "Test response from Gemini"

    def test_call_gemini_with_history(self):
        """Test Gemini API call with conversation history."""
        service = ChatbotService()

        # Add history
        service.memory.add_message("conv_123", "user", "Previous question")
        service.memory.add_message("conv_123", "assistant", "Previous response")

        # Mock the Gemini response
        mock_response = MagicMock()
        mock_response.text = "Context-aware response"

        mock_chat = MagicMock()
        mock_chat.send_message.return_value = mock_response

        service.gemini_model.start_chat.return_value = mock_chat

        result = service._call_gemini("Follow-up question", "conv_123")

        assert result == "Context-aware response"

        # Verify history was passed to start_chat
        call_kwargs = service.gemini_model.start_chat.call_args[1]
        assert 'history' in call_kwargs
        assert len(call_kwargs['history']) == 2

    def test_call_gemini_role_mapping(self):
        """Test that 'assistant' role is mapped to 'model' for Gemini API."""
        service = ChatbotService()

        # Add history with assistant role
        service.memory.add_message("conv_123", "user", "Question")
        service.memory.add_message("conv_123", "assistant", "Answer")

        # Mock the Gemini response
        mock_response = MagicMock()
        mock_response.text = "Mapped response"

        mock_chat = MagicMock()
        mock_chat.send_message.return_value = mock_response

        service.gemini_model.start_chat.return_value = mock_chat

        result = service._call_gemini("New question", "conv_123")

        # Verify role mapping in history
        call_kwargs = service.gemini_model.start_chat.call_args[1]
        history = call_kwargs['history']

        # User role should stay 'user'
        assert history[0]['role'] == 'user'
        # Assistant role should be mapped to 'model'
        assert history[1]['role'] == 'model'

    def test_call_gemini_empty_response(self):
        """Test handling of empty Gemini response (safety filter)."""
        service = ChatbotService()

        # Mock empty response
        mock_response = MagicMock()
        mock_response.text = None
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = MagicMock()

        mock_chat = MagicMock()
        mock_chat.send_message.return_value = mock_response

        service.gemini_model.start_chat.return_value = mock_chat

        result = service._call_gemini("Test message", "conv_123")

        # Should return a polite fallback message
        assert "couldn't generate a response" in result
        assert "safety filters" in result

    def test_call_gemini_with_model_fallback(self):
        """Test _call_gemini_with_model for model fallback."""
        service = ChatbotService()

        # Mock successful response from fallback model
        mock_response = MagicMock()
        mock_response.text = "Response from fallback model"

        with patch('google.generativeai.GenerativeModel') as mock_model_class:
            mock_model = MagicMock()
            mock_chat = MagicMock()
            mock_chat.send_message.return_value = mock_response
            mock_model.start_chat.return_value = mock_chat
            mock_model_class.return_value = mock_model

            result = service._call_gemini_with_model("Test message", "conv_123", "gemini-2.0-flash-lite")

            assert result == "Response from fallback model"

            # Verify model was created with system instruction
            mock_model_class.assert_called_once()
            call_kwargs = mock_model_class.call_args[1]
            assert call_kwargs['model_name'] == "gemini-2.0-flash-lite"
            assert 'system_instruction' in call_kwargs

    def test_call_gemini_with_model_quota_error(self):
        """Test _call_gemini_with_model raises exception on quota error."""
        service = ChatbotService()

        # Mock quota error
        with patch('google.generativeai.GenerativeModel') as mock_model_class:
            mock_model = MagicMock()
            mock_chat = MagicMock()
            mock_chat.send_message.side_effect = Exception("429 RESOURCE_EXHAUSTED")
            mock_model.start_chat.return_value = mock_chat
            mock_model_class.return_value = mock_model

            with pytest.raises(Exception, match="429 RESOURCE_EXHAUSTED"):
                service._call_gemini_with_model("Test", "conv_123", "gemini-2.5-flash")


# ============================================================================
# Model Fallback Chain Tests
# ============================================================================

class TestModelFallbackChain:
    """Tests for the model fallback chain on quota errors."""

    @patch('chatbot_module.backend.chatbot_service.ChatbotService._initialize_gemini')
    def _mock_gemini_init(self):
        """Mock Gemini initialization."""
        self.api_key_configured = True
        self.gemini_model = MagicMock()
        self._system_prompt = "Test system prompt"
        self._fallback_models = ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash"]

    def test_model_fallback_chain_on_quota_error(self):
        """Test that the service tries fallback models when primary hits quota limit."""
        service = ChatbotService()

        call_count = 0
        models_tried = []

        def mock_call_with_model(message, conv_id, model_name):
            nonlocal call_count, models_tried
            call_count += 1
            models_tried.append(model_name)

            if model_name == "gemini-2.5-flash":
                raise Exception("429 RESOURCE_EXHAUSTED")
            elif model_name == "gemini-2.0-flash-lite":
                raise Exception("429 RESOURCE_EXHAUSTED")
            else:
                return "Success from gemini-2.0-flash"

        service._call_gemini_with_model = mock_call_with_model

        result = service.chat("Test message")

        assert result["success"] is True
        assert result["model_used"] == "gemini-2.0-flash"
        assert call_count == 3
        assert models_tried == ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash"]

    def test_model_fallback_all_models_quota_exhausted(self):
        """Test that service falls back to fallback responses when all models hit quota."""
        service = ChatbotService()

        def mock_call_all_quota(message, conv_id, model_name):
            raise Exception("429 RESOURCE_EXHAUSTED")

        service._call_gemini_with_model = mock_call_all_quota

        result = service.chat("What does IC50 mean?")

        assert result["success"] is True
        assert result["model_used"] == "fallback"
        assert "IC50" in result["message"]

    def test_model_fallback_non_quota_error(self):
        """Test that non-quota errors don not trigger model fallback."""
        service = ChatbotService()

        def mock_call_non_quota(message, conv_id, model_name):
            raise Exception("API connection failed")

        service._call_gemini_with_model = mock_call_non_quota

        result = service.chat("Test message")

        # Should fall back to fallback response (not try other models)
        assert result["success"] is True
        assert result["model_used"] == "fallback"

    def test_model_fallback_switches_active_model(self):
        """Test that successful fallback switches the active model."""
        service = ChatbotService()

        call_count = 0

        def mock_call_primary_quota_then_success(message, conv_id, model_name):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # Primary model quota error
                raise Exception("429 RESOURCE_EXHAUSTED")
            else:
                # Fallback model success
                return "Response from fallback model"

        service._call_gemini_with_model = mock_call_primary_quota_then_success

        result = service.chat("Test message")

        assert result["success"] is True
        assert result["model_used"] == "gemini-2.0-flash-lite"
        # Active model should be updated
        assert service.gemini_model_name == "gemini-2.0-flash-lite"


# ============================================================================
# Context Enhancement Tests
# ============================================================================

class TestContextEnhancement:
    """Tests for context enhancement in chat messages."""

    @patch('chatbot_module.backend.chatbot_service.ChatbotService._initialize_gemini')
    def _mock_gemini_init(self):
        """Mock Gemini initialization."""
        self.api_key_configured = True
        self.gemini_model = MagicMock()

    def test_context_enhancement_with_prediction_data(self):
        """Test that prediction context is properly enhanced in the message."""
        service = ChatbotService()

        context = {
            "predicted_ic50": -1.46,
            "model_used": "CatBoost",
            "cell_line_name": "A172",
            "drug_name": "Camptothecin"
        }

        # Mock the Gemini call to capture the enhanced message
        enhanced_messages = []

        def capture_enhanced_message(message, conv_id, model_name):
            enhanced_messages.append(message)
            return "Enhanced response"

        service._call_gemini_with_model = capture_enhanced_message

        result = service.chat("What does this mean?", context=context)

        # Verify context was included in the enhanced message
        assert len(enhanced_messages) > 0
        enhanced_msg = enhanced_messages[0]
        assert "[Context:" in enhanced_msg
        assert "predicted_ic50: -1.46" in enhanced_msg
        assert "model_used: CatBoost" in enhanced_msg
        assert "cell_line_name: A172" in enhanced_msg
        assert "drug_name: Camptothecin" in enhanced_msg
        assert "User question: What does this mean?" in enhanced_msg

    def test_context_enhancement_without_context(self):
        """Test that messages without context are not enhanced."""
        service = ChatbotService()

        # Mock the Gemini call
        messages = []

        def capture_message(message, conv_id, model_name):
            messages.append(message)
            return "Simple response"

        service._call_gemini_with_model = capture_message

        result = service.chat("What does IC50 mean?")

        # Verify no context prefix in the message
        assert len(messages) > 0
        assert "[Context:" not in messages[0]
        assert messages[0] == "What does IC50 mean?"