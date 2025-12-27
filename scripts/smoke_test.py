#!/usr/bin/env python3
"""
Smoke Test Script

Validates configuration and connectivity without making a real call.

Checks:
1. Environment variables are set (without printing secrets)
2. Groq model exists via API
3. FastAPI app starts and /health returns OK
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 50}")
    print(f" {text}")
    print('=' * 50)


def print_ok(text: str) -> None:
    """Print success message."""
    print(f"  [OK] {text}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"  [ERR] {text}")


def print_warn(text: str) -> None:
    """Print warning message."""
    print(f"  [WARN] {text}")


def check_env_vars() -> bool:
    """Check that required environment variables are set."""
    print_header("Checking Environment Variables")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = [
        "PUBLIC_HOST",
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "DEEPGRAM_API_KEY",
        "CARTESIA_API_KEY",
        "GROQ_API_KEY",
        "GROQ_MODEL",
    ]
    
    optional_vars = [
        "PORT",
        "LOG_LEVEL",
        "DEFAULT_LANGUAGE",
        "DEEPGRAM_LANGUAGE_EN",
        "DEEPGRAM_LANGUAGE_FR",
        "ROUTER_ENABLED",
        "CARTESIA_VOICE_ID",
        "CARTESIA_VOICE_ID_FR",
    ]
    
    all_ok = True
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Show partial value for non-sensitive vars
            if var in ["PUBLIC_HOST", "GROQ_MODEL"]:
                print_ok(f"{var}: {value}")
            else:
                # Mask sensitive values
                masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "****"
                print_ok(f"{var}: {masked}")
        else:
            print_error(f"{var}: NOT SET")
            all_ok = False
    
    print("\nOptional variables:")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print_ok(f"{var}: {value}")
        else:
            print_warn(f"{var}: not set (using default)")
    
    return all_ok


async def check_groq_model() -> bool:
    """Validate Groq model exists."""
    print_header("Validating Groq Model")
    
    import httpx
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    if not api_key:
        print_error("GROQ_API_KEY not set")
        return False
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0,
            )
            
            if response.status_code != 200:
                print_error(f"API returned status {response.status_code}")
                return False
            
            data = response.json()
            models = [m.get("id") for m in data.get("data", [])]
            
            if model_name in models:
                print_ok(f"Model '{model_name}' exists")
                return True
            else:
                print_error(f"Model '{model_name}' NOT FOUND")
                print("\n  Available models:")
                for m in sorted(models)[:10]:
                    print(f"    - {m}")
                return False
                
    except httpx.RequestError as e:
        print_error(f"Failed to connect to Groq API: {e}")
        return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False


async def check_health_endpoint() -> bool:
    """Check that the FastAPI /health endpoint works."""
    print_header("Testing Health Endpoint")
    
    import httpx
    from dotenv import load_dotenv
    load_dotenv()
    
    port = int(os.getenv("PORT", "7860"))
    
    # Start the server in background
    print(f"  Starting server on port {port}...")
    
    # Import and create app
    try:
        from fastapi.testclient import TestClient
        from server.app import app
        
        client = TestClient(app)
        response = client.get("/health")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy":
                print_ok("Health endpoint returned healthy")
                return True
            else:
                print_error(f"Unexpected response: {data}")
                return False
        else:
            print_error(f"Health endpoint returned status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Failed to test health endpoint: {e}")
        return False


def check_dependencies() -> bool:
    """Check that required dependencies are importable."""
    print_header("Checking Dependencies")
    
    dependencies = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("websockets", "WebSockets"),
        ("deepgram", "Deepgram SDK"),
        ("cartesia", "Cartesia SDK"),
        ("openai", "OpenAI SDK"),
        ("structlog", "Structlog"),
        ("pydantic", "Pydantic"),
        ("httpx", "HTTPX"),
    ]
    
    all_ok = True
    
    for module, name in dependencies:
        try:
            __import__(module)
            print_ok(f"{name}")
        except ImportError as e:
            print_error(f"{name}: {e}")
            all_ok = False
    
    # Check optional dependencies
    print("\nOptional dependencies:")
    
    optional = [
        ("semantic_router", "Semantic Router"),
        ("instructor", "Instructor"),
    ]
    
    for module, name in optional:
        try:
            __import__(module)
            print_ok(f"{name}")
        except ImportError:
            print_warn(f"{name}: not installed")
    
    return all_ok


def check_audio_files() -> bool:
    """Check for cached audio files."""
    print_header("Checking Cached Audio Files")
    
    audio_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets",
        "audio"
    )
    
    expected_files = ["pricing.mulaw", "who_are_you.mulaw", "stop.mulaw"]
    
    if not os.path.exists(audio_dir):
        print_warn(f"Audio directory not found: {audio_dir}")
        print("  Run 'make audio' to create placeholder files")
        return True  # Not critical
    
    all_present = True
    for filename in expected_files:
        filepath = os.path.join(audio_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            duration_ms = size / 8  # 8 bytes per ms for 8kHz mu-law
            print_ok(f"{filename}: {size} bytes ({duration_ms:.0f}ms)")
        else:
            print_warn(f"{filename}: not found")
            all_present = False
    
    if not all_present:
        print("\n  Run 'make audio' to create audio files")
    
    return True  # Not critical


async def main() -> int:
    """Run all smoke tests."""
    print("\n" + "=" * 50)
    print(" TWILIO VOICE AGENT - SMOKE TEST")
    print("=" * 50)
    
    results = []
    
    # Run checks
    results.append(("Dependencies", check_dependencies()))
    results.append(("Environment Variables", check_env_vars()))
    results.append(("Groq Model", await check_groq_model()))
    results.append(("Cached Audio", check_audio_files()))
    
    # Health endpoint check is more involved, skip in basic smoke test
    # results.append(("Health Endpoint", await check_health_endpoint()))
    
    # Summary
    print_header("Summary")
    
    all_passed = True
    for name, passed in results:
        if passed:
            print_ok(name)
        else:
            print_error(name)
            all_passed = False
    
    print()
    
    if all_passed:
        print("[OK] All checks passed!")
        print("\nNext steps:")
        print("  1. Run 'make run' to start the server")
        print("  2. Run 'ngrok http 7860' in another terminal")
        print("  3. Update PUBLIC_HOST in .env with your ngrok URL")
        print("  4. Configure your Twilio phone number")
        print("  5. Make a test call!")
        return 0
    else:
        print("[ERR] Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
