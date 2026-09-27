"""A simple assistant agent built on the Claude Agent SDK."""

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
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
    )


async def run(prompt: str, verbose: bool = True) -> str | None:
    """Send one prompt to the agent, print progress, and return the final answer."""
    final: str | None = None
    async for message in query(prompt=prompt, options=build_options()):
        if isinstance(message, AssistantMessage) and verbose:
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Claude: {block.text}")
                elif isinstance(block, ToolUseBlock):
                    print(f"  -> tool {block.name}({block.input})")
        elif isinstance(message, ResultMessage):
            if message.is_error:
                raise RuntimeError(f"Agent failed: {message.errors or message.result}")
            final = message.result
            if verbose:
                cost = f"${message.total_cost_usd:.4f}" if message.total_cost_usd else "n/a"
                print(f"\n[done in {message.num_turns} turns, {message.duration_ms} ms, cost {cost}]")
    return final
