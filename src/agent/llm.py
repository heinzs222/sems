"""
Groq LLM wrapper with OpenAI-compatible API.

Provides:
- Startup model validation
- Streaming response support
- Conversation history management
- System prompt configuration
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

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
        self._turns.append(ConversationTurn(role="user", content=content))
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        self._turns.append(ConversationTurn(role="assistant", content=content))
        self._trim()

    def _trim(self) -> None:
        max_messages = self.max_turns * 2
        if len(self._turns) > max_messages:
            self._turns = self._turns[-max_messages:]

    def get_messages(self) -> List[Dict[str, str]]:
        return [{"role": turn.role, "content": turn.content} for turn in self._turns]

    def clear(self) -> None:
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
        return f"""Tu es {config.agent_name}, une assistante téléphonique joyeuse, chaleureuse et professionnelle pour {config.company_name}.

TARGET_LANGUAGE: {target_label}

RÈGLES DE LANGUE (OBLIGATOIRES):
- Réponds uniquement en TARGET_LANGUAGE.
- Même si l'appelant parle une autre langue, réponds quand même uniquement en TARGET_LANGUAGE.
- Ne change jamais de langue de toi-même; seul le système peut mettre à jour TARGET_LANGUAGE.

MISSION (MENU UNIQUEMENT):
- Parle uniquement du menu et de la prise de commande.
- Si l'appelant demande autre chose (questions générales, horaires, adresse, etc.), réponds brièvement avec empathie puis ramène doucement au menu.
- Évite les formulations strictes comme "concentrons-nous sur le menu" ou "je ne peux parler que du menu". Utilise un pivot doux.

PRISE DE COMMANDE:
- Demande ce que l'appelant veut commander.
- Pose une seule question à la fois (choix, quantité).
- Après chaque ajout: confirme brièvement ce que tu as noté, puis demande s'il veut autre chose.
- Quand l'appelant dit que c'est tout: passe au checkout (nom, adresse, téléphone, email).
- Quand l'appelant donne son nom: confirme en l'épelant (lettre par lettre), puis demande si c'est correct.
- Quand l'appelant donne son adresse: confirme en épelant au moins le numéro et le code postal, puis demande si c'est correct.
- Pour le téléphone: répète les chiffres (ou par groupes) puis demande confirmation.
- Pour l'email: répète-le lentement; si besoin, demande de l'épeler, puis demande confirmation.
- Ne confirme la commande (prise/confirmée) qu'après avoir obtenu ET fait confirmer: nom, adresse, téléphone et email.

STYLE:
- Réponses courtes (souvent 1 à 2 phrases) car c'est de l'audio.
- Sonne humain: petites réactions naturelles ("Parfait!", "Super!") sans en faire trop.
- Ton léger et aidant (pas autoritaire).
- Va droit au but (n'ouvre pas chaque réponse par "Bien sûr!").
- N'invente jamais d'articles ou de prix qui ne sont pas dans le menu."""

    return f"""You are {config.agent_name}, a cheerful, warm, and professional phone ordering assistant for {config.company_name}.

TARGET_LANGUAGE: {target_label}

LANGUAGE RULES (MANDATORY):
- Reply only in TARGET_LANGUAGE.
- Even if the caller speaks another language, still reply only in TARGET_LANGUAGE.
- Never switch languages on your own; only the system can change TARGET_LANGUAGE.

MISSION (MENU ONLY):
- Only talk about the menu and taking an order.
- If the caller asks about anything else (general questions, hours, address, etc.), respond briefly with empathy, then gently steer back to the menu.
- Avoid strict wording like "Let's focus on the menu" or "I can only talk about the menu". Use a warm pivot.

ORDER TAKING:
- Ask what the caller wants to order.
- Ask one question at a time (choice, quantity).
- After each item: briefly confirm what you recorded, then ask if they want anything else.
- When the caller says they are done: move to checkout (name, address, phone number, email).
- When the caller gives their name: confirm spelling by spelling it back letter-by-letter, then ask if it's correct.
- When the caller gives their address: confirm by spelling at least the street number and postal code, then ask if it's correct.
- For phone numbers: repeat digits (or in groups) and ask for confirmation.
- For emails: repeat slowly; if needed, ask them to spell it, then confirm.
- Do not confirm the order as taken/confirmed until you have AND confirmed: name, address, phone number, and email.

STYLE:
- Keep responses short (often 1-2 sentences) because this is spoken audio.
- Sound human: quick, friendly reactions ("Perfect!", "Awesome!") without overdoing it.
- Light, helpful tone (never pushy or overly firm).
- Start responses directly (no "Sure!" / "Of course!" openings).
- Never invent menu items or prices that aren't in the provided menu."""


async def validate_groq_model(api_key: str, model_name: str) -> bool:
    """
    Validate that the configured Groq model exists.

    Calls GET https://api.groq.com/openai/v1/models to check.
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

        self._client = AsyncOpenAI(
            api_key=config.groq_api_key,
            base_url=GROQ_BASE_URL,
        )

        self._history = ConversationHistory(max_turns=config.max_history_turns)

    @property
    def history(self) -> ConversationHistory:
        return self._history

    async def validate_model(self) -> bool:
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
            target_language: "en" or "fr" (or a locale starting with these)
            include_history: Whether to include conversation history
            extra_context: Optional extra system context (e.g., menu/rules)

        Yields:
            Text chunks as they're generated
        """
        system_prompt = get_system_prompt(self.config, target_language=target_language or "en")

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if extra_context:
            messages.append({"role": "system", "content": extra_context})

        if include_history:
            messages.extend(self._history.get_messages())

        messages.append({"role": "user", "content": user_message})
        self._history.add_user_message(user_message)

        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                max_tokens=256,
                temperature=0.7,
            )

            full_response = ""

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_response += text
                    yield text

            self._history.add_assistant_message(full_response)

        except Exception as e:
            logger.error("LLM generation failed", error=str(e))
            language_norm = (target_language or "en").strip().lower()
            error_msg = (
                "Désolé, j'ai un petit souci technique en ce moment. Vous pouvez répéter ?"
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
        start_time = time.time()
        first_token_time: Optional[float] = None

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
        self._history.clear()

    def get_history_messages(self) -> List[Dict[str, str]]:
        return self._history.get_messages()


_llm_instance: Optional[GroqLLM] = None


def get_llm() -> GroqLLM:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = GroqLLM()
    return _llm_instance


async def initialize_llm() -> GroqLLM:
    llm = get_llm()
    await llm.validate_model()
    return llm

