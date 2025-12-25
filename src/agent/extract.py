"""
Structured extraction using Instructor.

Extracts structured information from conversation turns:
- name: Caller's name if mentioned
- phone: Phone number if mentioned
- intent: What the caller wants
- next_step: Suggested next action
- consent_to_follow_up: Whether caller agreed to follow-up

Runs as a background task to avoid blocking the live audio loop.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Callable, Any
import json

import structlog
from pydantic import BaseModel, Field

from src.agent.config import get_config

logger = structlog.get_logger(__name__)

# Lazy import instructor to avoid startup overhead
_instructor_client = None


class CallExtraction(BaseModel):
    """Structured extraction from a conversation turn."""
    
    name: Optional[str] = Field(
        default=None,
        description="The caller's name if they mentioned it"
    )
    
    phone: Optional[str] = Field(
        default=None,
        description="A phone number if the caller mentioned one"
    )
    
    intent: str = Field(
        default="unknown",
        description="The caller's primary intent or reason for calling"
    )
    
    next_step: Optional[str] = Field(
        default=None,
        description="Suggested next action based on the conversation"
    )
    
    consent_to_follow_up: bool = Field(
        default=False,
        description="Whether the caller agreed to be contacted again"
    )


@dataclass
class ExtractionResult:
    """Result of an extraction attempt."""
    success: bool
    extraction: Optional[CallExtraction] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


def _get_instructor_client():
    """Lazy load the instructor client."""
    global _instructor_client
    
    if _instructor_client is not None:
        return _instructor_client
    
    try:
        import instructor
        from groq import Groq
        
        config = get_config()
        
        # Create Groq client with instructor patch
        groq_client = Groq(api_key=config.groq_api_key)
        _instructor_client = instructor.from_groq(groq_client, mode=instructor.Mode.JSON)
        
        logger.info("Instructor client initialized")
        return _instructor_client
        
    except Exception as e:
        logger.error("Failed to initialize instructor client", error=str(e))
        return None


async def extract_from_turn(
    user_message: str,
    assistant_message: str,
    conversation_history: Optional[List[dict]] = None,
    max_retries: int = 2,
) -> ExtractionResult:
    """
    Extract structured information from a conversation turn.
    
    Args:
        user_message: The user's message in this turn
        assistant_message: The assistant's response
        conversation_history: Optional previous turns for context
        max_retries: Number of retries on failure
        
    Returns:
        ExtractionResult with the extracted data or error
    """
    start_time = asyncio.get_event_loop().time()
    
    client = _get_instructor_client()
    if client is None:
        return ExtractionResult(
            success=False,
            error="Instructor client not initialized",
        )
    
    config = get_config()
    
    # Build context from history
    context = ""
    if conversation_history:
        context = "Previous conversation:\n"
        for turn in conversation_history[-3:]:  # Last 3 turns for context
            role = turn.get("role", "")
            content = turn.get("content", "")
            context += f"{role}: {content}\n"
    
    extraction_prompt = f"""Analyze this phone conversation turn and extract structured information.

{context}
Current turn:
User: {user_message}
Assistant: {assistant_message}

Extract the following if present:
- name: The caller's name (if they introduced themselves)
- phone: Any phone number mentioned
- intent: What the caller wants (be specific and concise)
- next_step: What should happen next based on the conversation
- consent_to_follow_up: Did the caller agree to be contacted again?

Only extract information that is explicitly stated or clearly implied."""

    for attempt in range(max_retries + 1):
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            extraction = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=config.groq_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an extraction assistant. Extract structured information from phone conversations accurately and concisely."
                        },
                        {
                            "role": "user",
                            "content": extraction_prompt
                        }
                    ],
                    response_model=CallExtraction,
                    max_retries=0,  # We handle retries ourselves
                )
            )
            
            end_time = asyncio.get_event_loop().time()
            latency_ms = (end_time - start_time) * 1000
            
            logger.info(
                "Extraction completed",
                intent=extraction.intent,
                has_name=extraction.name is not None,
                has_phone=extraction.phone is not None,
                latency_ms=round(latency_ms, 2),
            )
            
            return ExtractionResult(
                success=True,
                extraction=extraction,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            logger.warning(
                "Extraction attempt failed",
                attempt=attempt + 1,
                max_retries=max_retries,
                error=str(e),
            )
            
            if attempt < max_retries:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            
            end_time = asyncio.get_event_loop().time()
            return ExtractionResult(
                success=False,
                error=str(e),
                latency_ms=(end_time - start_time) * 1000,
            )
    
    return ExtractionResult(
        success=False,
        error="Max retries exceeded",
    )


class ExtractionQueue:
    """
    Background queue for extraction tasks.
    
    Runs extractions asynchronously without blocking the main audio loop.
    """
    
    def __init__(self, max_concurrent: int = 2):
        self.max_concurrent = max_concurrent
        self._queue: asyncio.Queue = asyncio.Queue()
        self._results: List[ExtractionResult] = []
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._callbacks: List[Callable[[ExtractionResult], Any]] = []
    
    def add_callback(self, callback: Callable[[ExtractionResult], Any]) -> None:
        """Add a callback to be called when extraction completes."""
        self._callbacks.append(callback)
    
    async def start(self) -> None:
        """Start the extraction worker."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._worker())
        logger.info("Extraction queue started")
    
    async def stop(self) -> None:
        """Stop the extraction worker."""
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("Extraction queue stopped")
    
    async def submit(
        self,
        user_message: str,
        assistant_message: str,
        conversation_history: Optional[List[dict]] = None,
    ) -> None:
        """
        Submit a turn for extraction.
        
        This is non-blocking and returns immediately.
        """
        await self._queue.put({
            "user_message": user_message,
            "assistant_message": assistant_message,
            "conversation_history": conversation_history,
        })
        
        logger.debug("Extraction task submitted", queue_size=self._queue.qsize())
    
    async def _worker(self) -> None:
        """Background worker that processes extraction tasks."""
        while self._running:
            try:
                # Wait for a task with timeout
                try:
                    task_data = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process with semaphore to limit concurrency
                async with self._semaphore:
                    result = await extract_from_turn(
                        user_message=task_data["user_message"],
                        assistant_message=task_data["assistant_message"],
                        conversation_history=task_data.get("conversation_history"),
                    )
                    
                    self._results.append(result)
                    
                    # Call callbacks
                    for callback in self._callbacks:
                        try:
                            callback(result)
                        except Exception as e:
                            logger.error("Extraction callback failed", error=str(e))
                    
                    self._queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Extraction worker error", error=str(e))
    
    def get_results(self) -> List[ExtractionResult]:
        """Get all extraction results."""
        return self._results.copy()
    
    def get_latest_extraction(self) -> Optional[CallExtraction]:
        """Get the most recent successful extraction."""
        for result in reversed(self._results):
            if result.success and result.extraction:
                return result.extraction
        return None
    
    def get_aggregated_data(self) -> dict:
        """
        Get aggregated data from all extractions.
        
        Merges extractions, preferring the most recent non-null values.
        """
        aggregated = {
            "name": None,
            "phone": None,
            "intents": [],
            "next_steps": [],
            "consent_to_follow_up": False,
        }
        
        for result in self._results:
            if not result.success or not result.extraction:
                continue
            
            ext = result.extraction
            
            if ext.name:
                aggregated["name"] = ext.name
            if ext.phone:
                aggregated["phone"] = ext.phone
            if ext.intent and ext.intent != "unknown":
                aggregated["intents"].append(ext.intent)
            if ext.next_step:
                aggregated["next_steps"].append(ext.next_step)
            if ext.consent_to_follow_up:
                aggregated["consent_to_follow_up"] = True
        
        return aggregated


# Optional: Outlines-based extraction (behind feature flag)
async def extract_with_outlines(
    user_message: str,
    assistant_message: str,
) -> ExtractionResult:
    """
    Extract using Outlines for constrained generation.
    
    This is an alternative to Instructor that guarantees valid JSON output.
    Only used when OUTLINES_ENABLED=true.
    """
    config = get_config()
    
    if not config.outlines_enabled:
        logger.warning("Outlines extraction called but OUTLINES_ENABLED=false")
        return ExtractionResult(
            success=False,
            error="Outlines not enabled",
        )
    
    try:
        # Lazy import
        import outlines
        from outlines import models, generate
        
        # This is a simplified example - in production you'd want to
        # cache the model and generator
        
        start_time = asyncio.get_event_loop().time()
        
        # Use a local model or API
        # Note: Outlines works best with local models for constrained generation
        logger.warning("Outlines extraction not fully implemented - using Instructor fallback")
        
        # Fall back to instructor
        return await extract_from_turn(user_message, assistant_message)
        
    except ImportError:
        logger.error("Outlines not installed")
        return ExtractionResult(
            success=False,
            error="Outlines not installed",
        )
    except Exception as e:
        logger.error("Outlines extraction failed", error=str(e))
        return ExtractionResult(
            success=False,
            error=str(e),
        )
