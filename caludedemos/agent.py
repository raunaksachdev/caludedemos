"""A simple assistant agent built on the Claude Agent SDK."""

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    StreamEvent,
    ToolUseBlock,
    query,
)

from caludedemos.config import MAX_TURNS, MODEL
from caludedemos.tools import HELPER_TOOL_NAMES, SERVER_NAME, helper_server

SYSTEM_PROMPT = (
    "You are a concise helper agent. Use the calculate tool for arithmetic and "
    "get_current_time for anything date/time related. You may read files in the "
    "working directory to answer questions about this project."
)


def build_options() -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        max_turns=MAX_TURNS,
        mcp_servers={SERVER_NAME: helper_server},
        # Restrict the agent to read-only built-ins plus our custom tools.
        tools=["Read", "Glob", "Grep"],
        allowed_tools=["Read", "Glob", "Grep", *HELPER_TOOL_NAMES],
        # Also yield raw API stream events so text can be printed as it is generated.
        include_partial_messages=True,
    )


async def run(prompt: str, verbose: bool = True) -> str | None:
    """Send one prompt to the agent, print progress, and return the final answer."""
    final: str | None = None
    in_text = False  # inside a streamed text block
    async for message in query(prompt=prompt, options=build_options()):
        if isinstance(message, StreamEvent) and verbose:
            # Text is printed here, delta by delta; the full AssistantMessage follows later.
            event = message.event
            if event["type"] == "content_block_start" and event["content_block"]["type"] == "text":
                in_text = True
                print("Claude: ", end="", flush=True)
            elif event["type"] == "content_block_delta" and event["delta"]["type"] == "text_delta":
                print(event["delta"]["text"], end="", flush=True)
            elif event["type"] == "content_block_stop" and in_text:
                in_text = False
                print(flush=True)
        elif isinstance(message, AssistantMessage) and verbose:
            # Text was already streamed above; tool calls print once their input is complete.
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    print(f"  -> tool {block.name}({block.input})")
        elif isinstance(message, ResultMessage):
            if message.is_error:
                raise RuntimeError(f"Agent failed: {message.errors or message.result}")
            final = message.result
            if verbose:
                cost = f"${message.total_cost_usd:.4f}" if message.total_cost_usd else "n/a"
                print(f"\n[done in {message.num_turns} turns, {message.duration_ms} ms, cost {cost}]")
    return final
