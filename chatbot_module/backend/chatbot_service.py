"""
Chatbot service for the Drug Response Prediction API.

This service provides:
- Gemini API integration for AI-powered responses
- Domain-specific system prompt for the drug response prediction project
- Conversation history management (via ConversationMemory)
- Graceful error handling for API failures
- Model fallback when primary model hits quota limits (429)
- Fallback responses when Gemini API is unavailable (via chatbot_utils)

The chatbot is designed for educational and explainability purposes only.
It does NOT provide medical diagnosis or treatment recommendations.

ORGANIZATION NOTE:
This file was extracted from api/services/chatbot_service.py and refactored
to use the modular components:
- chatbot_utils: Environment loading, fallback responses, topic matching
- chatbot_memory: Conversation history management
- chatbot_schema: Pydantic request/response models

The system prompt is loaded from prompts/system_prompt.txt at runtime.
If the file is not found, the embedded SYSTEM_PROMPT constant is used as fallback.
"""

import os
import logging
import uuid
from typing import Dict, List, Any, Optional
from pathlib import Path

from chatbot_module.backend.chatbot_utils import (
    FALLBACK_RESPONSES,
    match_fallback_topic,
)
from chatbot_module.backend.chatbot_memory import ConversationMemory

logger = logging.getLogger(__name__)


# ============================================================================
# SYSTEM PROMPT — Domain-specific context for the drug response project
# ============================================================================

# Try to load from prompts file; fall back to embedded constant
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

def _load_system_prompt() -> str:
    """
    Load the system prompt from prompts/system_prompt.txt.
    
    If the file doesn't exist, fall back to the embedded SYSTEM_PROMPT constant.
    
    Returns:
        The system prompt string
    """
    prompt_path = _PROMPTS_DIR / "system_prompt.txt"
    try:
        if prompt_path.exists():
            content = prompt_path.read_text(encoding="utf-8")
            logger.info(f"Loaded system prompt from {prompt_path}")
            return content
    except Exception as e:
        logger.warning(f"Failed to load system prompt from {prompt_path}: {e}")
    
    logger.info("Using embedded SYSTEM_PROMPT constant")
    return SYSTEM_PROMPT


# Embedded system prompt — used as fallback if prompts/system_prompt.txt is missing
SYSTEM_PROMPT = """You are an AI assistant for the Drug Response Prediction and Recommendation Platform — a graduation project in bioinformatics. Your role is to help users understand the system's predictions, methodology, and scientific concepts.

## YOUR CAPABILITIES

You can explain:
- **IC50 values**: What they mean, how they're predicted, and how to interpret them
- **Drug sensitivity**: How the system determines if a drug is effective for a cell line
- **Recommendation scores**: How drugs are ranked using score = 1/(1+IC50)
- **Preprocessing pipeline**: Label encoding, feature selection, and standard scaling
- **ML models**: CatBoost, XGBoost, LightGBM, GradientBoosting, Ridge, ElasticNet, Lasso
- **Project architecture**: How the prediction and recommendation engines work
- **Data sources**: GDSC dataset, cell line details, compound annotations

## PROJECT ARCHITECTURE KNOWLEDGE

### Preprocessing Pipeline (3 stages):
1. **Label Encoding**: 10 categorical columns (CELL_LINE_NAME, TCGA_DESC, DRUG_NAME, TARGET, TARGET_PATHWAY, SITE, HISTOLOGY, GDSC_TISSUE_DESCRIPTOR_1, GDSC_TISSUE_DESCRIPTOR_2, CANCER_TYPE_MATCHING_TCGA_LABEL) are encoded using saved LabelEncoder artifacts. Unseen labels are mapped to -1 (sentinel value).
2. **Feature Selection**: Exactly 9 features are selected — AUC, Z_SCORE, and 7 encoded columns (TARGET_ENC, TARGET_PATHWAY_ENC, TCGA_DESC_ENC, GDSC_TISSUE_DESCRIPTOR_2_ENC, CANCER_TYPE_MATCHING_TCGA_LABEL_ENC, SITE_ENC, GDSC_TISSUE_DESCRIPTOR_1_ENC). Note: CELL_LINE_NAME_ENC and DRUG_NAME_ENC are NOT in the final features — the model uses biological/pharmacological characteristics instead of raw identifiers.
3. **Standard Scaling**: The 9 selected features are normalized to zero mean and unit variance using a saved StandardScaler (x_scaled = (x - μ) / σ).

### Prediction Pipeline:
- Input → Preprocessing (load from disk, never fit at runtime) → Model prediction → IC50 output
- The preprocessor is ONLY loaded from saved artifacts in core/saved_preprocessor/
- Default model: CatBoost (best performing)

### Recommendation Engine:
- For a given cell line, predicts IC50 for ALL available drugs
- Ranks drugs by effectiveness: lower IC50 = better
- Computes recommendation score: score = 1 / (1 + IC50)
- Returns ranked list with drug name, IC50, score, target, and pathway

### IC50 Interpretation:
- IC50 (Half Maximal Inhibitory Concentration) measures drug effectiveness
- Lower IC50 = drug is more effective (needs less concentration to inhibit cell growth)
- LN_IC50 is the natural log of IC50, which is what our models predict
- IC50 < -2: Very sensitive (drug highly effective)
- IC50 between -2 and 0: Sensitive (drug likely effective)
- IC50 between 0 and 2: Moderate (average effectiveness)
- IC50 > 2: Resistant (drug likely not effective)

### Available Models (7 regression models):
- CatBoost (default, best performing)
- XGBoost
- LightGBM
- GradientBoosting
- ElasticNet
- Ridge
- Lasso

### Data Sources:
- GDSC_DATASET.csv: Main drug response data with cell line-drug combinations
- Cell_Lines_Details.xlsx: COSMIC tissue classification (SITE, HISTOLOGY)
- Compounds-annotation.csv: Drug information (targets, pathways)

## SAFETY RESTRICTIONS (MANDATORY)

- You MUST NOT provide medical diagnosis
- You MUST NOT prescribe treatments or drug recommendations for real patients
- You MUST NOT suggest that predictions should replace clinical decisions
- You MUST always clarify that this is an educational/research tool
- You MUST encourage users to consult healthcare professionals for medical decisions
- If a user asks about treating a real patient, redirect them to a qualified oncologist

## COMMUNICATION STYLE

- Be scientific but accessible — explain complex concepts clearly
- Use analogies when helpful (e.g., "IC50 is like a effectiveness score for drugs")
- Provide specific numbers and thresholds when discussing predictions
- Reference the project's methodology when explaining results
- Be concise but thorough — don't oversimplify, but don't overwhelm
- Use formatting (bullet points, bold text) to organize complex explanations
- When a user provides prediction context (IC50 value, cell line, drug), interpret it specifically

## RESPONSE FORMAT

When explaining a prediction result, include:
1. What the IC50 value means in plain language
2. How it compares to sensitivity thresholds
3. What biological factors the model considered
4. A reminder that this is for research/educational purposes only

When explaining a concept, include:
1. A clear definition
2. How it relates to the project
3. Practical implications
4. Any limitations or caveats"""


class ChatbotService:
    """
    Chatbot service for the Drug Response Prediction API.
    
    This service:
    - Connects to the Gemini API using google-generativeai
    - Applies a domain-specific system prompt
    - Manages conversation history per session (via ConversationMemory)
    - Provides fallback responses when Gemini is unavailable (via chatbot_utils)
    - Handles model fallback on quota errors (429)
    - Handles errors gracefully
    
    Attributes:
        gemini_model: The Gemini model instance (None if API key not configured)
        api_key_configured: Whether the Gemini API key is available
        gemini_model_name: The currently active Gemini model name
        memory: ConversationMemory instance for history management
    """
    
    def __init__(self):
        """Initialize the chatbot service."""
        self.gemini_model = None
        self.api_key_configured = False
        self.gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        # Fallback models to try if primary model hits quota limits (429)
        self._fallback_models = ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash"]
        self.memory = ConversationMemory()
        
        # Load the system prompt (from file or embedded constant)
        self._system_prompt = _load_system_prompt()
        
        self._initialize_gemini()
    
    def _initialize_gemini(self) -> None:
        """
        Initialize the Gemini API client.
        
        Attempts to configure the Gemini API with the key from environment.
        If the key is missing or invalid, the service falls back to pre-defined responses.
        """
        api_key = os.getenv("GEMINI_API_KEY", "")
        
        if not api_key or api_key == "your_gemini_api_key_here":
            logger.warning(
                "GEMINI_API_KEY not configured. Chatbot will use fallback responses. "
                "Set the key in .env file to enable Gemini-powered responses."
            )
            self.api_key_configured = False
            return
        
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=api_key)
            
            # Create the model with system instruction
            self.gemini_model = genai.GenerativeModel(
                model_name=self.gemini_model_name,
                system_instruction=self._system_prompt
            )
            
            self.api_key_configured = True
            logger.info(
                f"Chatbot service initialized with Gemini model: {self.gemini_model_name}"
            )
            
        except ImportError:
            logger.warning(
                "google-generativeai package not installed. "
                "Install it with: pip install google-generativeai. "
                "Chatbot will use fallback responses."
            )
            self.api_key_configured = False
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            self.api_key_configured = False
    
    def is_available(self) -> bool:
        """
        Check if the chatbot service is available (Gemini API configured and working).
        
        Returns:
            True if Gemini API is available, False otherwise
        """
        return self.api_key_configured and self.gemini_model is not None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get chatbot service status information.
        
        Returns:
            Dictionary with availability status, model name, and configuration info
        """
        return {
            "available": self.is_available(),
            "model": self.gemini_model_name if self.api_key_configured else None,
            "api_key_configured": self.api_key_configured,
            "message": (
                "Chatbot service is available with Gemini AI" 
                if self.is_available() 
                else "Chatbot service is in fallback mode (GEMINI_API_KEY not configured)"
            )
        }
    
    def chat(self, message: str, conversation_id: Optional[str] = None, 
             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a chat message and return an AI response.
        
        Args:
            message: User message
            conversation_id: Optional conversation ID for context continuity
            context: Optional context data (e.g., prediction results)
            
        Returns:
            Dictionary with response data including:
            - success: Whether the response was generated successfully
            - message: The AI response text
            - conversation_id: The conversation ID (new or existing)
            - model_used: The model that generated the response
        """
        # Generate or use existing conversation ID
        conv_id = conversation_id or str(uuid.uuid4())
        
        # Build context-enhanced message if context is provided
        enhanced_message = message
        if context:
            context_parts = []
            for key, value in context.items():
                context_parts.append(f"{key}: {value}")
            enhanced_message = f"[Context: {', '.join(context_parts)}]\n\nUser question: {message}"
        
        # Try Gemini API first (with model fallback on quota errors)
        if self.is_available():
            # Build list of models to try: primary first, then fallbacks
            models_to_try = [self.gemini_model_name]
            for fb in self._fallback_models:
                if fb not in models_to_try:
                    models_to_try.append(fb)
            
            for model_name in models_to_try:
                try:
                    response = self._call_gemini_with_model(enhanced_message, conv_id, model_name)
                    
                    # Store in conversation history
                    self.memory.add_message(conv_id, "user", message)
                    self.memory.add_message(conv_id, "assistant", response)
                    
                    # Update active model if fallback succeeded
                    if model_name != self.gemini_model_name:
                        logger.info(f"[CHAT] Switched active model from {self.gemini_model_name} to {model_name}")
                        self.gemini_model_name = model_name
                    
                    return {
                        "success": True,
                        "message": response,
                        "conversation_id": conv_id,
                        "model_used": model_name
                    }
                    
                except Exception as e:
                    error_str = str(e)
                    is_quota_error = "429" in error_str or "quota" in error_str.lower() or "RESOURCE_EXHAUSTED" in error_str
                    if is_quota_error and model_name != models_to_try[-1]:
                        logger.warning(f"[CHAT] Model {model_name} hit quota limit (429), trying next fallback model...")
                        continue
                    else:
                        logger.error(f"[CHAT] Gemini API call failed with {model_name}: {e}")
                        break
        
        # Use fallback response
        fallback_key = match_fallback_topic(message)
        fallback_message = FALLBACK_RESPONSES[fallback_key]
        
        # Store in conversation history
        self.memory.add_message(conv_id, "user", message)
        self.memory.add_message(conv_id, "assistant", fallback_message)
        
        return {
            "success": True,
            "message": fallback_message,
            "conversation_id": conv_id,
            "model_used": "fallback"
        }
    
    def _call_gemini(self, message: str, conversation_id: str) -> str:
        """
        Call the Gemini API to generate a response.
        
        Uses conversation history for context continuity if available.
        
        Args:
            message: Enhanced user message (with context if provided)
            conversation_id: Conversation ID for history lookup
            
        Returns:
            Generated response text from Gemini
            
        Raises:
            Exception: If the Gemini API call fails
        """
        # Get conversation history formatted for Gemini
        chat_history = self.memory.build_gemini_history(conversation_id)
        
        logger.info(f"[GEMINI_CALL] model={self.gemini_model_name}, conv_id={conversation_id}, history_len={len(chat_history)}, msg_len={len(message)}")
        
        chat_session = self.gemini_model.start_chat(history=chat_history)
        
        # Send message and get response
        logger.info("[GEMINI_CALL] Sending message to Gemini API...")
        response = chat_session.send_message(message)
        
        # Log raw response metadata
        logger.info(f"[GEMINI_CALL] Response received: candidates={len(response.candidates) if response.candidates else 0}")
        
        # Extract text from response
        if response.text:
            logger.info(f"[GEMINI_CALL] Response text extracted (len={len(response.text)})")
            return response.text
        else:
            # Empty response — check for safety filter blocks
            try:
                if response.candidates and response.candidates[0].finish_reason:
                    logger.warning(f"[GEMINI_CALL] Candidate finish_reason={response.candidates[0].finish_reason}")
            except Exception:
                pass
            logger.warning("[GEMINI_CALL] Gemini returned empty response (possibly filtered by safety settings)")
            return (
                "I apologize, but I couldn't generate a response for that query. "
                "This may be due to content safety filters. Please try rephrasing your question "
                "or ask about a different topic related to drug response prediction."
            )
    
    def _call_gemini_with_model(self, message: str, conversation_id: str, model_name: str) -> str:
        """
        Call the Gemini API with a specific model name.
        
        Used for model fallback when the primary model hits quota limits.
        
        Args:
            message: Enhanced user message (with context if provided)
            conversation_id: Conversation ID for history lookup
            model_name: The Gemini model name to use for this call
            
        Returns:
            Generated response text from Gemini
            
        Raises:
            Exception: If the Gemini API call fails
        """
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY", "")
        genai.configure(api_key=api_key)
        
        # Create a fresh model instance with the specified model name
        fallback_model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=self._system_prompt
        )
        
        # Get conversation history formatted for Gemini
        chat_history = self.memory.build_gemini_history(conversation_id)
        
        logger.info(f"[GEMINI_FALLBACK] Trying model={model_name}, conv_id={conversation_id}, history_len={len(chat_history)}")
        
        chat_session = fallback_model.start_chat(history=chat_history)
        
        # Send message and get response
        response = chat_session.send_message(message)
        
        # Extract text from response
        if response.text:
            logger.info(f"[GEMINI_FALLBACK] Model {model_name} responded successfully (len={len(response.text)})")
            return response.text
        else:
            logger.warning(f"[GEMINI_FALLBACK] Model {model_name} returned empty response")
            raise Exception(f"Model {model_name} returned empty response (possibly filtered)")
    
    def get_conversation_history(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get conversation history for a given conversation ID.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Dictionary with conversation history data
        """
        return self.memory.get_formatted_history(conversation_id)
    
    def clear_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Clear conversation history for a given conversation ID.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Dictionary confirming the conversation was cleared
        """
        return self.memory.clear_conversation(conversation_id)


# ============================================================================
# SINGLETON INSTANCE — Matches the pattern used by MLService
# ============================================================================

_chatbot_service_instance: Optional[ChatbotService] = None


def get_chatbot_service() -> ChatbotService:
    """
    Get or create the ChatbotService singleton.
    
    This follows the same singleton pattern as get_ml_service() in ml_service.py.
    
    Returns:
        ChatbotService instance
    """
    global _chatbot_service_instance
    if _chatbot_service_instance is None:
        _chatbot_service_instance = ChatbotService()
    return _chatbot_service_instance