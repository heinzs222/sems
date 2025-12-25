# =============================================================================
# Twilio Voice Agent - Docker Image
# =============================================================================
# Build: docker build -t twilio-voice-agent .
# Run:   docker run -p 7860:7860 --env-file .env twilio-voice-agent
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

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:7860/health'); r.raise_for_status()"

# Run the application with Granian (Rust-based ASGI server)
# Using /bin/sh -c to properly expand $PORT environment variable
CMD ["/bin/sh", "-c", "granian --interface asgi server.app:app --host 0.0.0.0 --port $PORT --log-level info"]
