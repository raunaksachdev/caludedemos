# caludedemos
This repository is to understand deep core claude SDK concepts and do POC work with claude

Built on the [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk) (Python), managed with [uv](https://docs.astral.sh/uv/).
The same helper agent is also implemented directly on the [Messages API](https://platform.claude.com/docs/en/api/messages) with a hand-written loop, for comparison.

| | Agent SDK (default, preferred) | Messages API (`--api`) |
|---|---|---|
| Agent loop | Run by the SDK (Claude Code harness) | Our own `for turn in ...` loop in `api_agent.py` |
| File tools | Built-in `Read` / `Glob` / `Grep` | Implemented in `api_tools.py` |
| Auth | API key, or your claude.ai login if no key is set | `ANTHROPIC_API_KEY` with Console credits |

## Setup

```bash
uv sync                 # create .venv and install dependencies
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env   # then replace with your real key
```

## Run

```bash
uv run main.py                                  # default demo task (Agent SDK)
uv run main.py "What is 17% of 2350?"           # your own task
uv run main.py --api "What is 17% of 2350?"     # same agent on the Messages API
uv run examples/01_hello_query.py               # minimal query() example
uv run examples/02_hello_messages_api.py        # minimal messages.create() example
uv run examples/03_interactive_client.py        # ClaudeSDKClient: interrupt (/stop) or message the agent mid-run
```

## Layout

| Path | Purpose |
|---|---|
| `main.py` | CLI entry point — sends a prompt to the helper agent (`--api` switches implementation) |
| `caludedemos/agent.py` | Agent SDK version: `ClaudeAgentOptions` and the `query()` loop |
| `caludedemos/tools.py` | Tool logic plus `@tool` wrappers exposed via an SDK MCP server |
| `caludedemos/api_agent.py` | Messages API version: hand-written agent loop |
| `caludedemos/api_tools.py` | Messages API tool schemas and handlers (incl. read-only file tools) |
| `caludedemos/config.py` | Model / max-turns settings loaded from `.env` |
| `examples/` | Standalone, numbered SDK concept demos |
