"""Medical Diagnostics Agent - A Bindu AI Agent for Medical Analysis."""

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, cast

from bindu.penguin.bindufy import bindufy
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .agents import run_medical_diagnosis

# Load environment variables from .env file
load_dotenv()

# Global agent instance
agent: Any = None
_initialized = False
_init_lock = asyncio.Lock()

# Setup logging
_logger = logging.getLogger(__name__)


def load_config() -> dict[str, Any]:
    """Load agent config from `agent_config.json` or return defaults."""
    config_path = Path(__file__).parent / "agent_config.json"

    if config_path.exists():
        try:
            with open(config_path) as f:
                return cast(dict[str, Any], json.load(f))
        except (OSError, json.JSONDecodeError) as exc:
            _logger.warning("Failed to load config from %s", config_path, exc_info=exc)

    return {
        "name": "medical-diagnostics-agent",
        "description": "AI-powered multi-specialist medical diagnostics assistant",
        "deployment": {
            "url": "http://127.0.0.1:3773",
            "expose": True,
            "protocol_version": "1.0.0",
        },
    }


class MedicalDiagnosticsAgent:
    """Medical Diagnostics Agent wrapper following the research-agent pattern."""

    def __init__(self, model_name: str = "gpt-4o"):
        """Initialize medical diagnostics agent with model name."""
        self.model_name = model_name

        # Model selection logic (only supports OpenRouter)
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        if openrouter_api_key:
            # Use OpenRouter API key with OpenAI client
            self.model = ChatOpenAI(
                model_name=model_name,
                openai_api_key=SecretStr(openrouter_api_key),
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0,
            )
            print(f"✅ Using OpenRouter model: {model_name}")
        else:
            # Define error message separately to avoid TRY003
            error_msg = (
                "No API key provided. Set OPENROUTER_API_KEY environment variable.\n"
                "For OpenRouter: https://openrouter.ai/keys"
            )
            raise ValueError(error_msg)

    async def arun(self, messages: list[dict[str, str]]) -> str:
        """Run the agent with the given messages - matches research-agent pattern."""
        # Extract medical report from messages
        medical_report = ""
        for message in messages:
            if message.get("role") == "user":
                medical_report = message.get("content", "")
                break

        if not medical_report:
            return "Error: No medical report provided in the user message."

        try:
            # Run the medical diagnosis pipeline
            final_diagnosis = await run_medical_diagnosis(medical_report, self.model_name)

            # Format the response
            response = f"""### Final Diagnosis:

{final_diagnosis}

---
*Analysis performed by AI Medical Diagnostics Agent*
*Specialist inputs: Cardiologist, Psychologist, Pulmonologist*
*Model: {self.model_name}*
"""

            return response

        except Exception as e:
            error_msg = f"Error during medical diagnosis: {e!s}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return error_msg


async def initialize_agent() -> None:
    """Initialize medical diagnostics agent with proper model."""
    global agent

    # Get API keys from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("MODEL_NAME", "openai/gpt-4o")

    # Model selection logic
    if openrouter_api_key:
        agent = MedicalDiagnosticsAgent(model_name)
        print(f"✅ Using OpenRouter model: {model_name}")
    elif openai_api_key:
        agent = MedicalDiagnosticsAgent()
        print("✅ Using OpenAI model: gpt-4o")
    else:
        # Define error message separately to avoid TRY003
        error_msg = (
            "No API key provided. Set OPENAI_API_KEY or OPENROUTER_API_KEY environment variable.\n"
            "For OpenAI: https://platform.openai.com/api-keys\n"
            "For OpenRouter: https://openrouter.ai/keys"
        )
        raise ValueError(error_msg)

    print("✅ Medical Diagnostics Agent initialized")


async def run_agent(messages: list[dict[str, str]]) -> Any:
    """Run the agent with the given messages - matches research-agent pattern."""
    global agent
    if not agent:
        # Define error message separately to avoid TRY003
        error_msg = "Agent not initialized"
        raise RuntimeError(error_msg)

    # Run the agent and get response - matches research-agent pattern
    return await agent.arun(messages)


async def handler(messages: list[dict[str, str]]) -> Any:
    """Handle incoming agent messages with lazy initialization - matches research-agent pattern."""
    global _initialized

    # Lazy initialization on first call
    async with _init_lock:
        if not _initialized:
            print("🔧 Initializing Medical Diagnostics Agent...")
            await initialize_agent()
            _initialized = True

    # Run the async agent
    result = await run_agent(messages)
    return result


async def cleanup() -> None:
    """Clean up any resources."""
    print("🧹 Cleaning up Medical Diagnostics Agent resources...")


def main():
    """Run the main entry point for the Medical Diagnostics Agent."""
    parser = argparse.ArgumentParser(description="Bindu Medical Diagnostics Agent")
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (env: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "gpt-4o"),
        help="Model name to use (env: MODEL_NAME, default: gpt-4o)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to agent_config.json (optional)",
    )
    args = parser.parse_args()

    # Set environment variables if provided via CLI
    if args.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_api_key
    if args.model:
        os.environ["MODEL_NAME"] = args.model

    print("🤖 Medical Diagnostics Agent - AI Multi-Specialist Health Analysis")
    print("🏥 Capabilities: Cardiac, Psychological, and Respiratory analysis")
    print("👥 Specialist Team: Cardiologist, Psychologist, Pulmonologist")

    # Load configuration
    config = load_config()

    try:
        # Bindufy and start the agent server
        print("🚀 Starting Bindu Medical Diagnostics Agent server...")
        print(f"🌐 Server will run on: {config.get('deployment', {}).get('url', 'http://127.0.0.1:3774')}")
        bindufy(config, handler)
    except KeyboardInterrupt:
        print("\n🛑 Medical Diagnostics Agent stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup on exit
        asyncio.run(cleanup())


if __name__ == "__main__":
    main()
