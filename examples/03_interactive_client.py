"""Interactive session with ClaudeSDKClient: talk to the agent while it is working.

Unlike query(), ClaudeSDKClient keeps one conversation open, so you can send
messages at any time and interrupt a run in progress.

    uv run examples/03_interactive_client.py

Type while the agent is idle -> starts a new request (same conversation).
Type while it is busy       -> message is sent without stopping it (see when Claude picks it up).
/stop                       -> interrupt the current run; the session and context stay open.
/quit (or Ctrl+D)           -> end the session.
"""

import sys

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    StreamEvent,
    ToolUseBlock,
)
from dotenv import load_dotenv

load_dotenv()

OPTIONS = ClaudeAgentOptions(
    system_prompt="You are a concise assistant exploring the project in the working directory.",
    tools=["Read", "Glob", "Grep"],  # read-only built-ins
    allowed_tools=["Read", "Glob", "Grep"],
    include_partial_messages=True,  # stream text as it is generated
)

busy = False  # True from sending a request until its ResultMessage arrives


async def print_messages(client: ClaudeSDKClient) -> None:
    """Print everything the session emits, for the whole session (not just one response)."""
    global busy
    in_text = False
    async for message in client.receive_messages():
        if isinstance(message, StreamEvent):
            event = message.event
            if event["type"] == "content_block_start" and event["content_block"]["type"] == "text":
                in_text = True
                print("Claude: ", end="", flush=True)
            elif event["type"] == "content_block_delta" and event["delta"]["type"] == "text_delta":
                print(event["delta"]["text"], end="", flush=True)
            elif event["type"] == "content_block_stop" and in_text:
                in_text = False
                print(flush=True)
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    print(f"  -> tool {block.name}({block.input})")
        elif isinstance(message, ResultMessage):
            busy = False
            print(f"[run ended: {message.subtype}, {message.num_turns} turns, "
                  f"{message.duration_ms} ms] — type a message, /stop or /quit")


async def main() -> None:
    global busy
    print(__doc__)
    async with ClaudeSDKClient(options=OPTIONS) as client:
        async with anyio.create_task_group() as tg:
            tg.start_soon(print_messages, client)
            while True:
                # Read input in a thread so the printer keeps streaming meanwhile.
                line = await anyio.to_thread.run_sync(sys.stdin.readline, abandon_on_cancel=True)
                text = line.strip()
                if not line or text == "/quit":
                    if busy:
                        await client.interrupt()
                    break
                if text == "/stop":
                    if busy:
                        print("[interrupting…]")
                        await client.interrupt()
                    else:
                        print("[nothing running]")
                elif text:
                    print(f"[{'sent while busy' if busy else 'sent'}] You: {text}")
                    busy = True
                    await client.query(text)
            tg.cancel_scope.cancel()


anyio.run(main)
