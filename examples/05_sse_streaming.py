"""Server-Sent Events (SSE) streaming on the Messages API.

The API streams a response as SSE events: message_start, then per content block
content_block_start / content_block_delta... / content_block_stop, then
message_delta (stop_reason + usage) and message_stop. The SDK parses the SSE
wire format for us; `messages.stream()` also accumulates the events so
`get_final_message()` returns the complete Message at the end.

This demo renders thinking summaries and answer text live as deltas arrive,
tallies the SSE event types, and checks stop_reason on the final message.
Streaming is the recommended default for long outputs / high max_tokens: it
avoids HTTP timeouts on requests that run for minutes.

Usage:
    uv run examples/05_sse_streaming.py
    uv run examples/05_sse_streaming.py "Explain how TCP congestion control works"
"""

import sys
from collections import Counter

import anthropic
from dotenv import load_dotenv

load_dotenv()

DEFAULT_PROMPT = "Explain, step by step, why the sky is blue and why sunsets are red."
DIM, RESET = "\033[2m", "\033[0m"
# Raw SSE event types; the SDK helper also yields convenience events
# ("text", "thinking", ...) derived from these, which we don't tally.
SSE_EVENT_TYPES = {
    "message_start", "content_block_start", "content_block_delta",
    "content_block_stop", "message_delta", "message_stop",
}

prompt = " ".join(sys.argv[1:]) or DEFAULT_PROMPT
client = anthropic.Anthropic()
event_counts: Counter[str] = Counter()

try:
    with client.beta.messages.stream(
        model="claude-opus-5-5",
        max_tokens=64000,  # streaming means a large cap won't hit HTTP timeouts
        # Thinking is always on for Opus 5.5; "summarized" streams readable
        # summaries instead of empty thinking blocks (a long silent pause).
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": "medium"},
        # If a safety classifier declines, the API re-runs the request on a
        # fallback model inside the same stream.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for event in stream:
            if event.type in SSE_EVENT_TYPES:
                event_counts[event.type] += 1

            if event.type == "message_start":
                print(f"{DIM}[message_start id={event.message.id}]{RESET}")
            elif event.type == "content_block_start":
                block_type = event.content_block.type
                if block_type == "thinking":
                    print(f"\n{DIM}[thinking]{RESET}")
                elif block_type == "text":
                    print("\n[answer]")
                elif block_type == "fallback":
                    print(f"\n{DIM}[fallback: switched model mid-stream]{RESET}")
            elif event.type == "content_block_delta":
                if event.delta.type == "thinking_delta":
                    print(f"{DIM}{event.delta.thinking}{RESET}", end="", flush=True)
                elif event.delta.type == "text_delta":
                    print(event.delta.text, end="", flush=True)
            elif event.type == "message_delta":
                print(f"\n{DIM}[message_delta stop_reason={event.delta.stop_reason}]{RESET}")

        message = stream.get_final_message()
except anthropic.RateLimitError as e:
    sys.exit(f"\nRate limited; retry after {e.response.headers.get('retry-after', '?')}s.")
except anthropic.APIStatusError as e:
    sys.exit(f"\nAPI error {e.status_code}: {e.message}")
except anthropic.APIConnectionError:
    # The stream dropped mid-response; any text printed so far is partial.
    sys.exit("\nConnection lost mid-stream. Please retry.")

if message.stop_reason == "refusal":
    category = message.stop_details.category if message.stop_details else None
    print(f"\nRequest declined (category={category}).")
elif message.stop_reason == "max_tokens":
    print("\nOutput hit max_tokens and is truncated.")

print(f"\n{DIM}--- SSE events received ---{RESET}")
for event_type, count in event_counts.items():
    print(f"{DIM}{event_type:<22}{count}{RESET}")
print(
    f"{DIM}served by {message.model}: {message.usage.input_tokens} input / "
    f"{message.usage.output_tokens} output tokens{RESET}"
)
