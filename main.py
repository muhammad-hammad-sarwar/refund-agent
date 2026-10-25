"""Entry point for the refund agent with conversation loop."""

import sys
import os
from pathlib import Path
import uuid
from dotenv import load_dotenv
load_dotenv()

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent.agent import create_agent, run_conversation_loop


def main():
    """Main entry point - runs the interactive conversation loop."""
    # Generate a new thread id
    thread_id = uuid.uuid4()

    # Get model from env or use default
    model_name = os.getenv("GROK_MODEL", "grok-beta")

    print("Starting Refund Agent...")
    print(f"Model: {model_name}")

    # Create agent
    agent = create_agent(
        thread_id=thread_id,
        model_name=model_name,
        temperature=0.1,
    )

    # Run conversation loop
    run_conversation_loop(agent)


if __name__ == "__main__":
    main()