"""
FastAPI server for Twilio Voice Agent.

Endpoints:
- GET /health: Health check
- GET /metrics: JSON metrics
- POST /twiml: Generate TwiML for Twilio webhook
- WS /ws: Twilio Media Streams WebSocket
"""

import asyncio
import sys

# 2025 Performance: Use uvloop for faster asyncio (Linux only)
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass  # uvloop not available on Windows

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import JSONResponse
import structlog
import uvicorn

# Configure logging before imports
from src.agent.config import get_config, init_config, ConfigError

# Initialize structured logging
def configure_logging(log_level: str = "INFO") -> None:
    """Configure structured logging."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if log_level != "DEBUG" else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Set log level
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

logger = structlog.get_logger(__name__)


@dataclass
class ServerMetrics:
    """Server-wide metrics."""
    start_time: float = field(default_factory=time.time)
    total_connections: int = 0
    active_connections: int = 0
    total_calls: int = 0
    active_calls: int = 0
    errors: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uptime_seconds": round(time.time() - self.start_time, 2),
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "total_calls": self.total_calls,
            "active_calls": self.active_calls,
            "errors": self.errors,
        }


# Global metrics
metrics = ServerMetrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Twilio Voice Agent server...")
    
    try:
        # Initialize and validate configuration
        config = init_config()
        configure_logging(config.log_level)
        
        # Validate Groq model at startup
        from src.agent.llm import initialize_llm
        await initialize_llm()
        
        # Initialize semantic router (lazy, but warm up)
        if config.router_enabled:
            from src.agent.routing import initialize_router
            initialize_router()
        
        logger.info(
            "Server ready",
            port=config.port,
            public_host=config.public_host,
            ws_url=config.ws_url,
        )
        
    except ConfigError as e:
        logger.error("Configuration error", error=str(e))
        sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        logger.error("Startup failed", error=str(e))
        sys.exit(1)
    
    yield
    
    # Shutdown
    logger.info("Shutting down server...")


# Create FastAPI app
app = FastAPI(
    title="Twilio Voice Agent",
    description="AI-powered voice agent for Twilio phone calls",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse(
        content={
            "status": "healthy",
            "timestamp": time.time(),
            "active_calls": metrics.active_calls,
        }
    )


@app.get("/metrics")
async def get_metrics() -> JSONResponse:
    """Metrics endpoint."""
    return JSONResponse(content=metrics.to_dict())


@app.post("/twiml")
@app.get("/twiml")
@app.post("/incoming-call")
@app.get("/incoming-call")
async def generate_twiml(request: Request) -> Response:
    """
    Generate TwiML for Twilio webhook.
    
    Returns TwiML that connects to our WebSocket endpoint.
    """
    config = get_config()
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{config.public_host}/ws" />
    </Connect>
</Response>"""
    
    logger.info("Generated TwiML", ws_url=f"wss://{config.public_host}/ws")
    
    return Response(
        content=twiml,
        media_type="application/xml",
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Twilio Media Streams WebSocket endpoint.
    
    Handles incoming audio and sends outgoing audio for a call.
    """
    await websocket.accept()
    
    metrics.total_connections += 1
    metrics.active_connections += 1
    metrics.total_calls += 1
    metrics.active_calls += 1
    
    call_id = f"call_{int(time.time() * 1000)}"
    
    logger.info(
        "WebSocket connected",
        call_id=call_id,
        active_calls=metrics.active_calls,
    )
    
    # Import here to avoid circular imports and speed up startup
    from src.agent.pipeline import create_pipeline
    
    pipeline = None
    
    async def send_message(message: str) -> None:
        """Send a message to the WebSocket."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error("Failed to send WebSocket message", error=str(e))
    
    try:
        # Create and start pipeline
        pipeline = await create_pipeline(send_message)
        
        # Handle incoming messages
        while True:
            try:
                message = await websocket.receive_text()
                await pipeline.handle_message(message)
                
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected", call_id=call_id)
                break
            except Exception as e:
                logger.error(
                    "Error handling WebSocket message",
                    call_id=call_id,
                    error=str(e),
                )
                metrics.errors += 1
                # Continue processing - don't crash on single message error
                continue
                
    except Exception as e:
        logger.error(
            "WebSocket handler error",
            call_id=call_id,
            error=str(e),
        )
        metrics.errors += 1
        
    finally:
        # Cleanup
        if pipeline:
            try:
                await pipeline.stop()
            except Exception as e:
                logger.error("Error stopping pipeline", error=str(e))
        
        metrics.active_connections -= 1
        metrics.active_calls -= 1
        
        logger.info(
            "Call ended",
            call_id=call_id,
            active_calls=metrics.active_calls,
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        error=str(exc),
    )
    metrics.errors += 1
    
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


def main() -> None:
    """Run the server."""
    try:
        config = get_config()
    except Exception:
        # Use defaults if config fails
        config = type('Config', (), {'port': 7860, 'log_level': 'INFO'})()
    
    configure_logging(getattr(config, 'log_level', 'INFO'))
    
    logger.info(
        "Starting server",
        port=getattr(config, 'port', 7860),
    )
    
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=getattr(config, 'port', 7860),
        log_level=getattr(config, 'log_level', 'INFO').lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
