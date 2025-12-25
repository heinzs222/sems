# =============================================================================
# Twilio Voice Agent - Docker Image (2025 Performance Stack)
# =============================================================================
# Build: docker build -t twilio-voice-agent .
# Run:   docker run -p 8080:8080 --env-file .env twilio-voice-agent
# Updated: 2025-12-25 - Using granian for low-latency WebSocket
# =============================================================================

FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p assets/audio assets/audio/source logs \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port (Railway assigns $PORT dynamically)
EXPOSE 8080

# Health check disabled - Railway handles this
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD python -c "import httpx; r = httpx.get('http://localhost:8080/health'); r.raise_for_status()"

# Run with granian (2025 performance stack)
CMD ["granian", "--interface", "asgi", "server.app:app", "--host", "0.0.0.0", "--port", "8080", "--websockets-impl", "websockets"]
