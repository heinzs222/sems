.PHONY: install install-uv install-pip run test audio lint clean help

# Default Python version
PYTHON ?= python3

# Detect OS for platform-specific commands
ifeq ($(OS),Windows_NT)
    VENV_ACTIVATE = .venv\Scripts\activate
    VENV_PYTHON = .venv\Scripts\python
    RM_CMD = rmdir /s /q
    MKDIR_CMD = mkdir
else
    VENV_ACTIVATE = .venv/bin/activate
    VENV_PYTHON = .venv/bin/python
    RM_CMD = rm -rf
    MKDIR_CMD = mkdir -p
endif

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## Install dependencies (tries uv first, falls back to pip)
	@echo "Installing dependencies..."
	@if command -v uv >/dev/null 2>&1; then \
		echo "Using uv..."; \
		uv venv .venv; \
		uv pip install -r requirements.txt; \
	else \
		echo "uv not found, using pip..."; \
		$(PYTHON) -m venv .venv; \
		$(VENV_PYTHON) -m pip install --upgrade pip; \
		$(VENV_PYTHON) -m pip install -r requirements.txt; \
	fi
	@echo ""
	@echo "Installation complete!"
	@echo "Activate your virtual environment:"
	@echo "  source .venv/bin/activate  (Linux/Mac)"
	@echo "  .venv\\Scripts\\activate     (Windows)"

install-uv: ## Install dependencies using uv (faster)
	@echo "Installing with uv..."
	uv venv .venv
	uv pip install -r requirements.txt

install-pip: ## Install dependencies using pip
	@echo "Installing with pip..."
	$(PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt

run: ## Run the server
	@echo "Starting Twilio Voice Agent server..."
	@if [ -f .env ]; then \
		$(VENV_PYTHON) -m server.app; \
	else \
		echo "ERROR: .env file not found!"; \
		echo "Copy .env.example to .env and fill in your credentials."; \
		exit 1; \
	fi

run-dev: ## Run the server in development mode with auto-reload
	@echo "Starting server in development mode..."
	$(VENV_PYTHON) -m uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

test: ## Run tests
	@echo "Running tests..."
	$(VENV_PYTHON) -m pytest tests/ -v

test-cov: ## Run tests with coverage
	@echo "Running tests with coverage..."
	$(VENV_PYTHON) -m pytest tests/ -v --cov=src --cov-report=html

smoke: ## Run smoke test to verify configuration
	@echo "Running smoke test..."
	$(VENV_PYTHON) scripts/smoke_test.py

audio: ## Build cached audio files from WAV sources
	@echo "Building cached audio files..."
	@if [ ! -d "assets/audio" ]; then $(MKDIR_CMD) assets/audio; fi
	@chmod +x scripts/build_audio.sh 2>/dev/null || true
	@bash scripts/build_audio.sh

lint: ## Run linting
	@echo "Running linting..."
	$(VENV_PYTHON) -m ruff check src/ server/ tests/

format: ## Format code
	@echo "Formatting code..."
	$(VENV_PYTHON) -m ruff format src/ server/ tests/

clean: ## Clean up generated files
	@echo "Cleaning up..."
	$(RM_CMD) __pycache__ 2>/dev/null || true
	$(RM_CMD) .pytest_cache 2>/dev/null || true
	$(RM_CMD) .ruff_cache 2>/dev/null || true
	$(RM_CMD) htmlcov 2>/dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleanup complete!"

docker-build: ## Build Docker image
	docker build -t twilio-voice-agent .

docker-run: ## Run Docker container
	docker run -p 7860:7860 --env-file .env twilio-voice-agent
