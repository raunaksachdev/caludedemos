"""The same helper agent, built directly on the Messages API with a hand-written loop.

Alternative to agent.py (Claude Agent SDK, the preferred option). Here our code
owns every turn: it calls the API, runs the tools Claude asks for, and decides
when to stop. Requires ANTHROPIC_API_KEY with Console credits.
"""

import time

import anthropic

from caludedemos.api_tools import TOOLS, execute_tool
from caludedemos.config import MAX_TURNS, MODEL

SYSTEM_PROMPT = (
    "You are a concise helper agent. Use the calculate tool for arithmetic and "
    "get_current_time for anything date/time related. You may read files in the "
    "working directory (read_file, glob, grep) to answer questions about this project."
)


async def run(prompt: str, verbose: bool = True) -> str | None:
    """Send one prompt through our own agent loop, print progress, and return the final answer."""
    client = anthropic.AsyncAnthropic()
    messages: list[dict] = [{"role": "user", "content": prompt}]
    input_tokens = output_tokens = 0
    start = time.monotonic()

    for turn in range(1, MAX_TURNS + 1):
        response = await client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
            # On a safety refusal, re-run the request on Anthropic's recommended fallback model.
            betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default"},
        )
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens

        if response.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined the request: {response.stop_details}")

        # Keep the full content (thinking + tool_use blocks), not just the text.
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "text" and verbose:
                print(f"Claude: {block.text}")
            elif block.type == "tool_use":
                if verbose:
                    print(f"  -> tool {block.name}({block.input})")
                result, is_error = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                    "is_error": is_error,
                })

        if response.stop_reason == "max_tokens":
            raise RuntimeError("Response hit max_tokens before finishing")

        if response.stop_reason != "tool_use":
            if verbose:
                ms = int((time.monotonic() - start) * 1000)
                print(f"\n[done in {turn} turns, {ms} ms, "
                      f"{input_tokens} input / {output_tokens} output tokens]")
            return next((b.text for b in reversed(response.content) if b.type == "text"), None)

        # All results for this turn go back together in one user message.
        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"Agent stopped after reaching MAX_TURNS={MAX_TURNS}")
