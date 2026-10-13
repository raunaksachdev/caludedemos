"""Prompt caching on the Messages API: one large shared prefix, many questions.

Caching is a prefix match. The request renders tools -> system -> messages, and
a cache_control breakpoint caches everything up to and including its block. Any
byte change before the breakpoint is a miss.

This demo puts this repo's source files in the system prompt (the stable,
expensive part), marks it with cache_control, then asks several different
questions after it:

  1. first question  -> cache WRITE  (cache_creation_input_tokens, ~1.25x price)
  2. second question -> cache READ   (cache_read_input_tokens, ~0.05x on Opus 5.5)
  3. third question  -> cache READ
  4. same prefix, but with a timestamp prepended to the system prompt
                     -> cache MISS: a silent invalidator re-writes everything

Requires ANTHROPIC_API_KEY with Console credits (prompt caching is a Messages
API feature; the claude.ai login used by 06_agent_sdk_streaming.py can't call it).

Usage:
    uv run examples/07_prompt_caching.py
"""

import sys
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-5-5"
# Opus 5.5 prices per million tokens: input, cache write (5-min TTL, 1.25x), cache read, output.
PRICE_INPUT, PRICE_WRITE, PRICE_READ, PRICE_OUTPUT = 4.00, 5.00, 0.20, 20.00
DIM, RESET = "\033[2m", "\033[0m"

ROOT = Path(__file__).resolve().parent.parent
# sorted() keeps the prefix byte-identical between runs; glob order is not guaranteed.
SOURCE_FILES = sorted([ROOT / "README.md", *ROOT.glob("examples/*.py"), *ROOT.glob("caludedemos/*.py")])
REPO_CONTEXT = "\n\n".join(
    f'<file path="{path.relative_to(ROOT)}">\n{path.read_text()}\n</file>' for path in SOURCE_FILES
)
SYSTEM_PROMPT = (
    "You answer questions about the Python repository below. Be concise: at most "
    "three sentences, and name the files you are referring to.\n\n" + REPO_CONTEXT
)

QUESTIONS = [
    "Which examples need Console API credits and which can run on a claude.ai login?",
    "How does the hand-written agent loop in the Messages API version decide when to stop?",
    "Which example demonstrates human approval, and how does it pause for it?",
]

client = anthropic.Anthropic()


def ask(question: str, system_text: str) -> anthropic.types.beta.BetaMessage:
    """Ask one question after the shared prefix and return the final message."""
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=16000,
        # Thinking and effort are rendered into the prompt: changing them between
        # requests invalidates the cache, so pin them for every request.
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=[
            {
                "type": "text",
                "text": system_text,
                # Breakpoint at the end of the shared part. The question below varies
                # per request, so it goes after the breakpoint and is never cached.
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": question}],
    ) as stream:
        return stream.get_final_message()


def report(label: str, message: anthropic.types.beta.BetaMessage) -> None:
    usage = message.usage
    written = usage.cache_creation_input_tokens or 0
    read = usage.cache_read_input_tokens or 0
    # input_tokens is only the uncached remainder; the full prompt is the sum of all three.
    total_input = usage.input_tokens + written + read
    cost = (usage.input_tokens * PRICE_INPUT + written * PRICE_WRITE
            + read * PRICE_READ + usage.output_tokens * PRICE_OUTPUT) / 1e6
    uncached_cost = (total_input * PRICE_INPUT + usage.output_tokens * PRICE_OUTPUT) / 1e6
    status = "READ" if read else "WRITE" if written else "none"

    answer = next((block.text for block in message.content if block.type == "text"), "")
    if message.stop_reason == "refusal":
        answer = "(declined)"
    print(f"\n[{label}] cache {status}")
    print(answer.strip())
    print(f"{DIM}  prompt {total_input:>6} tokens = {usage.input_tokens} uncached + "
          f"{written} written + {read} read; output {usage.output_tokens}")
    print(f"  cost ${cost:.4f} (without caching: ${uncached_cost:.4f}){RESET}")


try:
    for i, question in enumerate(QUESTIONS, start=1):
        report(f"{i}. {question}", ask(question, SYSTEM_PROMPT))

    # A silent invalidator: a timestamp at the front of the system prompt changes the
    # prefix bytes, so the whole prefix is re-written instead of read. Put dynamic
    # values after the breakpoint (e.g. in the user message) instead.
    stamped = f"Current time: {datetime.now().isoformat()}\n\n{SYSTEM_PROMPT}"
    report(f"{len(QUESTIONS) + 1}. same question as 1, timestamp in system prompt",
           ask(QUESTIONS[0], stamped))
except anthropic.RateLimitError as e:
    sys.exit(f"\nRate limited; retry after {e.response.headers.get('retry-after', '?')}s.")
except anthropic.APIStatusError as e:
    sys.exit(f"\nAPI error {e.status_code}: {e.message}")
except anthropic.APIConnectionError:
    sys.exit("\nConnection error. Please retry.")
