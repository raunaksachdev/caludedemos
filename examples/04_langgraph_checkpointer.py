"""LangGraph + Claude: a support desk where a manager approves every reply.

Scenario: customers of "Mugly" (an online mug shop) open support tickets. Claude
drafts each reply, but nothing is sent until a manager approves it. Managers can
reject a draft with a note, approvals can arrive after a server restart, and every
step is kept as an audit trail.

    uv run examples/04_langgraph_checkpointer.py

The graph (one run = one customer message):

    customer message -> write_draft -> manager_approval --approve--> send_reply -> done
                            ^                |
                            +----reject------+

LangGraph concepts, in the order the demo shows them:
  Step 1  interrupt() pauses the run for the manager; Command(resume=...) continues it
  Step 2  the checkpointer remembers each ticket's conversation (thread_id = ticket id)
  Step 3  checkpoints live in SQLite, so a paused ticket survives a server restart
  Step 4  get_state_history() is a full audit trail of every step
  Step 5  time travel: go back to an old checkpoint, change a decision, replay from there

Claude is called through the Agent SDK (works with your claude.ai login). Each call is
stateless; the conversation Claude sees comes entirely from LangGraph's saved state.
"""

import operator
import os
from pathlib import Path
from typing import Annotated, TypedDict

import anyio
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

load_dotenv()

MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5")
DB_PATH = Path(__file__).parent / "support_tickets.db"
SUPPORT_POLICY = (
    "You are a support agent for Mugly, an online shop that sells mugs. "
    "Policy: damaged items get a free replacement, shipped within 2 business days. "
    "Write only the next support reply: at most two short sentences, plain text, no greeting line."
)


# ---------------------------------------------------------------- the graph

class Ticket(TypedDict):
    conversation: Annotated[list[str], operator.add]  # reducer: new lines are appended
    draft: str             # reply waiting for the manager
    manager_note: str      # why the last draft was rejected
    approved: bool


async def ask_claude(prompt: str) -> str:
    options = ClaudeAgentOptions(model=MODEL, system_prompt=SUPPORT_POLICY, tools=[], max_turns=1)
    reply = ""
    async for message in query(prompt=prompt, options=options):  # let the stream finish; don't break out early
        if isinstance(message, ResultMessage):
            reply = message.result or ""
    return reply


async def write_draft(ticket: Ticket) -> dict:
    prompt = "\n".join(ticket["conversation"])
    if ticket["manager_note"]:
        prompt += f"\n\n(Your previous draft was rejected by your manager: {ticket['manager_note']})"
    return {"draft": await ask_claude(prompt)}


def manager_approval(ticket: Ticket) -> dict:
    # The run stops here and is saved. It continues when someone calls
    # graph.ainvoke(Command(resume={...})); that value is what interrupt() returns.
    decision = interrupt(ticket["draft"])
    return {"approved": decision["approve"], "manager_note": decision.get("note", "")}


def send_reply(ticket: Ticket) -> dict:
    return {"conversation": [f"Support: {ticket['draft']}"], "manager_note": ""}


def after_approval(ticket: Ticket) -> str:
    return "send_reply" if ticket["approved"] else "write_draft"


def build_graph(checkpointer):
    graph = StateGraph(Ticket)
    graph.add_node("write_draft", write_draft)
    graph.add_node("manager_approval", manager_approval)
    graph.add_node("send_reply", send_reply)
    graph.add_edge(START, "write_draft")
    graph.add_edge("write_draft", "manager_approval")
    graph.add_conditional_edges("manager_approval", after_approval, ["send_reply", "write_draft"])
    graph.add_edge("send_reply", END)
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------- the people using it

def ticket(ticket_id: str) -> dict:
    return {"configurable": {"thread_id": ticket_id}}


async def customer_says(graph, ticket_id: str, text: str) -> None:
    print(f"\n  [{ticket_id}] Customer: {text}")
    new_message = {"conversation": [f"Customer: {text}"], "approved": False, "manager_note": ""}
    await graph.ainvoke(new_message, ticket(ticket_id))  # runs until manager_approval pauses it
    await show_status(graph, ticket_id)


async def manager(graph, ticket_id: str, approve: bool, note: str = "") -> None:
    print(f"  [{ticket_id}] Manager: " + ("APPROVE" if approve else f"REJECT - {note}"))
    await graph.ainvoke(Command(resume={"approve": approve, "note": note}), ticket(ticket_id))
    await show_status(graph, ticket_id)


async def show_status(graph, ticket_id: str) -> None:
    state = await graph.aget_state(ticket(ticket_id))
    if state.next == ("manager_approval",):
        print(f"  [{ticket_id}] PAUSED, draft waiting for manager: {state.values['draft']}")
    else:
        print(f"  [{ticket_id}] SENT: {state.values['conversation'][-1]}")


def step(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------- the demo

async def main() -> None:
    DB_PATH.unlink(missing_ok=True)

    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        graph = build_graph(saver)

        step("Step 1 - Human in the loop: the manager rejects a draft, then approves")
        await customer_says(graph, "T-1001", "My order #4521 arrived with a cracked mug. I want my money back.")
        await manager(graph, "T-1001", approve=False, note="Offer a free replacement, not a refund.")
        await manager(graph, "T-1001", approve=True)

        step("Step 2 - Memory: each ticket (thread_id) keeps its own conversation")
        await customer_says(graph, "T-1001", "OK, a replacement is fine. When will it ship?")
        await manager(graph, "T-1001", approve=True)
        # A different ticket starts empty; it knows nothing about order #4521.
        await customer_says(graph, "T-1002", "When will my replacement ship?")
        await manager(graph, "T-1002", approve=True)

        step("Step 3 - Durability: a ticket is waiting for approval when the server restarts")
        await customer_says(graph, "T-1003", "Do your mugs survive the dishwasher?")

    print("\n  ... server restarts: the in-process graph is gone, only support_tickets.db remains ...")

    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        graph = build_graph(saver)  # brand-new graph object, same database
        await show_status(graph, "T-1003")
        await manager(graph, "T-1003", approve=True)

        step("Step 4 - Audit trail: every step of ticket T-1001 was checkpointed")
        history = [snap async for snap in graph.aget_state_history(ticket("T-1001"))]
        for snap in reversed(history):  # history comes newest first; print oldest first
            next_node = snap.next[0] if snap.next else "(done)"
            print(f"  step {snap.metadata['step']:>2} | next: {next_node:<16} | draft: {snap.values.get('draft', '')[:50]}")

        step("Step 5 - Time travel: what if the manager had approved the FIRST draft?")
        first_pause = next(s for s in reversed(history) if s.next == ("manager_approval",))
        print(f"  going back to step {first_pause.metadata['step']}, draft was: {first_pause.values['draft']}")
        # Record a different manager decision on top of that old checkpoint, then continue from it.
        fork = await graph.aupdate_state(first_pause.config, {"approved": True}, as_node="manager_approval")
        await graph.ainvoke(None, fork)
        print("  the replayed timeline of T-1001:")
        for line in (await graph.aget_state(ticket("T-1001"))).values["conversation"]:
            print(f"    {line}")
        print("  (the original timeline is still in the history; the ticket now points at the new one)")


anyio.run(main)
