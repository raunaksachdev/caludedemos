"""CLI entry point: uv run main.py [--api] "your task here"

Uses the Claude Agent SDK by default; pass --api to run the same agent on the
Messages API with our own loop instead.
"""

import sys

import anyio

DEFAULT_PROMPT = (
    "What time is it right now, what is (1234 * 56) / 7, "
    "and which Python dependencies does pyproject.toml declare?"
)


def main() -> None:
    args = sys.argv[1:]
    use_api = "--api" in args
    if use_api:
        args.remove("--api")
        from caludedemos.api_agent import run
    else:
        from caludedemos.agent import run

    prompt = " ".join(args) or DEFAULT_PROMPT
    print(f"[{'Messages API' if use_api else 'Agent SDK'}] You: {prompt}\n")
    anyio.run(run, prompt)


if __name__ == "__main__":
    main()
