"""Agent SDK streaming on your claude.ai login (no API key, no Console credits).

Counterpart to 05_sse_streaming.py. With include_partial_messages=True the
Agent SDK yields a StreamEvent for every raw API stream event, interleaved with
the usual AssistantMessage / ResultMessage, so thinking summaries, answer text
and tool-call inputs can be printed while Claude is still generating them.

Auth: the bundled Claude Code CLI prefers ANTHROPIC_API_KEY when it is set, so
this script deliberately does not load .env and strips any key from the
environment. The CLI then falls back to your claude.ai login (`claude` /login),
and usage counts against your subscription.

Usage:
    uv run examples/06_agent_sdk_streaming.py
    uv run examples/06_agent_sdk_streaming.py "What does main.py do?"
"""

import os
import sys

import anyio
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    query,
)

DEFAULT_PROMPT = "Look at the examples/ folder and summarize in one line each what each example demonstrates."
DIM, RESET = "\033[2m", "\033[0m"

# Force the claude.ai login instead of an API key from the shell environment.
for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
    os.environ.pop(var, None)


async def main(prompt: str) -> None:
    options = ClaudeAgentOptions(
        # Read-only built-ins, so the agent can explore but not modify the repo.
        tools=["Read", "Glob", "Grep"],
        allowed_tools=["Read", "Glob", "Grep"],
        max_turns=10,
        # Isolate the demo from your personal ~/.claude settings, plugins and MCP
        # servers; the claude.ai login is stored separately and still applies.
        setting_sources=[],
        strict_mcp_config=True,
        # "summarized" streams readable thinking instead of empty thinking blocks.
        thinking={"type": "adaptive", "display": "summarized"},
        include_partial_messages=True,
    )

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, SystemMessage) and message.subtype == "init":
            # "none" means no key was used, i.e. the claude.ai login is in effect.
            print(f"{DIM}[init model={message.data.get('model')} "
                  f"apiKeySource={message.data.get('apiKeySource')}]{RESET}")

        elif isinstance(message, StreamEvent):
            event = message.event  # raw API stream event, as a dict
            if event["type"] == "content_block_start":
                block = event["content_block"]
                if block["type"] == "thinking":
                    print(f"\n{DIM}[thinking]{RESET}")
                elif block["type"] == "text":
                    print("\n[answer]")
                elif block["type"] == "tool_use":
                    print(f"\n{DIM}[tool {block['name']}] ", end="", flush=True)
            elif event["type"] == "content_block_delta":
                delta = event["delta"]
                if delta["type"] == "thinking_delta":
                    print(f"{DIM}{delta['thinking']}{RESET}", end="", flush=True)
                elif delta["type"] == "text_delta":
                    print(delta["text"], end="", flush=True)
                elif delta["type"] == "input_json_delta":
                    # Tool input arrives as JSON fragments while it is generated.
                    print(delta["partial_json"], end="", flush=True)
            elif event["type"] == "content_block_stop":
                print(RESET, end="", flush=True)

        elif isinstance(message, ResultMessage):
            if message.is_error:
                sys.exit(f"\nAgent failed: {message.errors or message.result}")
            print(f"\n\n{DIM}[done in {message.num_turns} turns, {message.duration_ms} ms, "
                  f"stop_reason={message.stop_reason}]{RESET}")


anyio.run(main, " ".join(sys.argv[1:]) or DEFAULT_PROMPT)
