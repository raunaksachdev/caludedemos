"""Smallest possible Agent SDK call: one prompt, no tools, print the result."""

import anyio
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
from dotenv import load_dotenv

load_dotenv()


async def main() -> None:
    options = ClaudeAgentOptions(tools=[], max_turns=1)
    async for message in query(prompt="Explain what an AI agent is in one sentence.", options=options):
        if isinstance(message, ResultMessage):
            print(message.result)


anyio.run(main)
