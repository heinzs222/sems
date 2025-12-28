"""
Groq LLM wrapper with OpenAI-compatible API.

Provides:
- Startup model validation
- Streaming response support
- Conversation history management
- System prompt configuration
"""

import asyncio
from typing import AsyncGenerator, List, Dict, Optional, Any
from dataclasses import dataclass, field
import time

import httpx
import structlog
from openai import AsyncOpenAI

from src.agent.config import get_config

logger = structlog.get_logger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


@dataclass
class LLMResponse:
    """Response from LLM."""
    text: str
    first_token_ms: float = 0.0
    total_ms: float = 0.0
    tokens_generated: int = 0


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


class ConversationHistory:
    """Manages conversation history with a rolling window."""
    
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._turns: List[ConversationTurn] = []
    
    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self._turns.append(ConversationTurn(role="user", content=content))
        self._trim()
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message."""
        self._turns.append(ConversationTurn(role="assistant", content=content))
        self._trim()
    
    def _trim(self) -> None:
        """Trim history to max turns."""
        # Keep pairs of turns (user + assistant)
        max_messages = self.max_turns * 2
        if len(self._turns) > max_messages:
            self._turns = self._turns[-max_messages:]
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get messages in OpenAI format."""
        return [
            {"role": turn.role, "content": turn.content}
            for turn in self._turns
        ]
    
    def clear(self) -> None:
        """Clear conversation history."""
        self._turns.clear()
    
    def __len__(self) -> int:
        return len(self._turns)


def get_system_prompt(config: Optional[Any] = None, target_language: str = "en") -> str:
    """
    Get the system prompt for the voice agent.
    
    This defines the agent's persona and behavior guidelines.
    """
    if config is None:
        config = get_config()
    
    language_norm = (target_language or "en").strip().lower()
    is_french = language_norm.startswith("fr")
    target_label = "French" if is_french else "English"

    if is_french:
        return f"""Tu es {config.agent_name}, une assistante téléphonique sympathique et professionnelle pour {config.company_name}.

TARGET_LANGUAGE: {target_label}

RÈGLES DE LANGUE (OBLIGATOIRES):
- Réponds uniquement en TARGET_LANGUAGE.
- Même si l'appelant parle une autre langue, réponds quand même uniquement en TARGET_LANGUAGE.
- Ne change jamais de langue de toi-même ; seul le système peut mettre à jour TARGET_LANGUAGE.

COMPORTEMENT:
- Sois naturelle et conversationnelle (appel téléphonique).
- Réponses courtes et claires (souvent 1 à 3 phrases) car c'est de l'audio.
- Si tu ne comprends pas, pose une question de clarification.

SÉCURITÉ ET CONFIDENTIALITÉ:
- Ne partage jamais d'informations sensibles sur d'autres appelants.
- Respecte la vie privée.

STYLE:
- Va droit au but (évite \"Bien sûr !\" au début).
- Utilise un ton chaleureux et professionnel.
- Évite les listes longues et les explications trop techniques."""

    return f"""You are {config.agent_name}, a friendly and helpful AI phone assistant for {config.company_name}.

TARGET_LANGUAGE: {target_label}

LANGUAGE RULES (MANDATORY):
- Reply only in TARGET_LANGUAGE.
- Even if the caller speaks another language, still reply only in TARGET_LANGUAGE.
- Never switch languages on your own; only the system can change TARGET_LANGUAGE.

CORE BEHAVIORS:
- Be conversational and natural - you're on a phone call
- Keep responses concise (1-3 sentences typically) - this is spoken audio
- Be warm and professional
- If you don't understand something, ask for clarification
- Never reveal that you're an AI unless directly asked

PHONE CALL GUIDELINES:
- Speak naturally as if in a real conversation
- Avoid long lists or complex information
- Use simple, clear language
- If the caller seems confused, offer to explain differently
- Be patient with interruptions - they're normal in phone calls

SAFETY AND COMPLIANCE:
- Never share sensitive information about other callers
- Don't make promises you can't keep
- If asked about something outside your knowledge, be honest
- Respect caller privacy
- If someone seems distressed, be empathetic and suggest appropriate resources

RESPONSE STYLE:
- Start responses directly - no "Sure!" or "Of course!"
- Use contractions (I'm, you're, we'll) for natural speech
- Avoid bullet points or numbered lists
- Keep technical terms simple"""


async def validate_groq_model(api_key: str, model_name: str) -> bool:
    """
    Validate that the configured Groq model exists.
    
    Calls GET https://api.groq.com/openai/v1/models to check.
    
    Args:
        api_key: Groq API key
        model_name: Model name to validate
        
    Returns:
        True if model exists
        
    Raises:
        SystemExit: If model doesn't exist (fail fast)
    """
    logger.info("Validating Groq model", model=model_name)
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{GROQ_BASE_URL}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0,
            )
            
            if response.status_code != 200:
                logger.error(
                    "Failed to fetch Groq models",
                    status_code=response.status_code,
                    response=response.text[:200],
                )
                raise SystemExit(
                    f"Failed to validate Groq model. API returned status {response.status_code}. "
                    "Check your GROQ_API_KEY."
                )
            
            data = response.json()
            models = data.get("data", [])
            model_ids = [m.get("id") for m in models]
            
            if model_name not in model_ids:
                available = ", ".join(sorted(model_ids)[:10])
                logger.error(
                    "Groq model not found",
                    requested_model=model_name,
                    available_models=available,
                )
                raise SystemExit(
                    f"GROQ_MODEL '{model_name}' not found in available models.\n"
                    f"Available models include: {available}\n"
                    "Please update GROQ_MODEL in your .env file."
                )
            
            logger.info("Groq model validated successfully", model=model_name)
            return True
            
        except httpx.RequestError as e:
            logger.error("Failed to connect to Groq API", error=str(e))
            raise SystemExit(
                f"Failed to connect to Groq API: {e}\n"
                "Check your network connection and GROQ_API_KEY."
            )


class GroqLLM:
    """
    Groq LLM client with streaming support.
    
    Uses OpenAI-compatible API for streaming responses.
    """
    
    def __init__(self, config: Optional[Any] = None):
        if config is None:
            config = get_config()
        
        self.config = config
        self.model = config.groq_model
        
        # Use OpenAI client with Groq base URL
        self._client = AsyncOpenAI(
            api_key=config.groq_api_key,
            base_url=GROQ_BASE_URL,
        )
        
        self._history = ConversationHistory(max_turns=config.max_history_turns)
    
    @property
    def history(self) -> ConversationHistory:
        """Get conversation history."""
        return self._history
    
    async def validate_model(self) -> bool:
        """Validate the configured model exists."""
        return await validate_groq_model(self.config.groq_api_key, self.model)
    
    async def generate_streaming(
        self,
        user_message: str,
        target_language: Optional[str] = None,
        include_history: bool = True,
        extra_context: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response.
        
        Args:
            user_message: The user's message
            include_history: Whether to include conversation history
            extra_context: Optional extra system context (e.g., menu/rules)
            
        Yields:
            Text chunks as they're generated
        """
        system_prompt = get_system_prompt(self.config, target_language=target_language or "en")

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        if extra_context:
            messages.append({"role": "system", "content": extra_context})
        
        if include_history:
            messages.extend(self._history.get_messages())
        
        messages.append({"role": "user", "content": user_message})
        
        # Add user message to history
        self._history.add_user_message(user_message)
        
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                max_tokens=256,  # Keep responses short for voice
                temperature=0.7,
            )
            
            full_response = ""
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_response += text
                    yield text
            
            # Add assistant response to history
            self._history.add_assistant_message(full_response)
            
        except Exception as e:
            logger.error("LLM generation failed", error=str(e))
            language_norm = (target_language or "en").strip().lower()
            error_msg = (
                "Désolé, j'ai un problème technique en ce moment. Pouvez-vous répéter ?"
                if language_norm.startswith("fr")
                else "I'm sorry, I'm having trouble right now. Could you please repeat that?"
            )
            self._history.add_assistant_message(error_msg)
            yield error_msg
    
    async def generate(
        self,
        user_message: str,
        target_language: Optional[str] = None,
        include_history: bool = True,
        extra_context: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a complete response (non-streaming).
        
        Args:
            user_message: The user's message
            include_history: Whether to include conversation history
            extra_context: Optional extra system context (e.g., menu/rules)
            
        Returns:
            LLMResponse with full text and timing
        """
        start_time = time.time()
        first_token_time = None
        
        full_text = ""
        token_count = 0
        
        async for chunk in self.generate_streaming(
            user_message,
            target_language=target_language,
            include_history=include_history,
            extra_context=extra_context,
        ):
            if first_token_time is None:
                first_token_time = time.time()
            full_text += chunk
            token_count += 1
        
        end_time = time.time()
        
        return LLMResponse(
            text=full_text,
            first_token_ms=(first_token_time - start_time) * 1000 if first_token_time else 0,
            total_ms=(end_time - start_time) * 1000,
            tokens_generated=token_count,
        )
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()
    
    def get_history_messages(self) -> List[Dict[str, str]]:
        """Get conversation history as list of messages."""
        return self._history.get_messages()


# Singleton instance
_llm_instance: Optional[GroqLLM] = None


def get_llm() -> GroqLLM:
    """Get or create the LLM singleton."""
    global _llm_instance
    
    if _llm_instance is None:
        _llm_instance = GroqLLM()
    
    return _llm_instance


async def initialize_llm() -> GroqLLM:
    """
    Initialize and validate the LLM at startup.
    
    Returns:
        Initialized and validated GroqLLM instance
    """
    llm = get_llm()
    await llm.validate_model()
    return llm
