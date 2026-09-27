"""Custom in-process tools exposed to the agent through an SDK MCP server."""

from datetime import datetime
from typing import Annotated, Any

from claude_agent_sdk import create_sdk_mcp_server, tool


# Plain implementations, shared with the Messages API agent (api_tools.py).
def current_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def evaluate(expression: str) -> str:
    """Evaluate arithmetic; raises ValueError on anything else."""
    if not set(expression) <= set("0123456789+-*/(). "):
        raise ValueError("unsupported characters")
    return str(eval(expression, {"__builtins__": {}}))  # restricted to arithmetic above


@tool("get_current_time", "Get the current local date and time.", {})
async def get_current_time(args: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": current_time()}]}


@tool(
    "calculate",
    "Evaluate a basic arithmetic expression using + - * / ( ) and numbers.",
    {"expression": Annotated[str, "Arithmetic expression, e.g. '(3 + 4) * 12'"]},
)
async def calculate(args: dict[str, Any]) -> dict[str, Any]:
    try:
        result = evaluate(args["expression"])
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return {"content": [{"type": "text", "text": f"Error: {e}"}], "is_error": True}
    return {"content": [{"type": "text", "text": result}]}


SERVER_NAME = "helpers"

helper_server = create_sdk_mcp_server(
    name=SERVER_NAME, tools=[get_current_time, calculate]
)

# MCP tools are addressed as mcp__<server>__<tool>
HELPER_TOOL_NAMES = [f"mcp__{SERVER_NAME}__{t.name}" for t in (get_current_time, calculate)]
