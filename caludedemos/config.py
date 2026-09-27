"""Environment-driven settings for the demo agents."""

import os

from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5")
MAX_TURNS = int(os.getenv("CLAUDE_MAX_TURNS", "10"))
