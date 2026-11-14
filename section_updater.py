"""
Exercise 1: Section Updater.

A minimal LangGraph agent that edits ## sections of notes.md based on a
natural-language request. Built to see exactly how StateGraph + tool-calling
work end to end: agent decides to call a tool -> tool runs -> result flows
back into state -> agent responds.
"""

import os
from typing import Annotated

from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

load_dotenv(override=True)

NOTES_PATH = os.path.join(os.path.dirname(__file__), "notes.md")


@tool
def update_section(section_name: str, new_content: str) -> str:
    """Replace the contents of a markdown '## section_name' section in notes.md with new_content."""
    with open(NOTES_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    header = f"## {section_name}"
    start = text.find(header)
    if start == -1:
        return f"Section '{section_name}' not found in notes.md"

    body_start = start + len(header)
    next_header = text.find("\n## ", body_start)
    body_end = next_header if next_header != -1 else len(text)

    updated = text[:body_start] + "\n\n" + new_content.strip() + "\n\n" + text[body_end:].lstrip("\n")

    with open(NOTES_PATH, "w", encoding="utf-8") as f:
        f.write(updated.rstrip() + "\n")

    return f"Updated section '{section_name}'."


@tool
def read_section(section_name: str) -> str:
    """Return the current contents of a markdown '## section_name' section in notes.md."""
    with open(NOTES_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    header = f"## {section_name}"
    start = text.find(header)
    if start == -1:
        return f"Section '{section_name}' not found in notes.md"

    body_start = start + len(header)
    next_header = text.find("\n## ", body_start)
    body_end = next_header if next_header != -1 else len(text)

    content = text[body_start:body_end].strip()
    return content if content else "(empty)"


TOOLS = [update_section, read_section]


class State(TypedDict):
    messages: Annotated[list, add_messages]
    approved: bool
    last_tool_call: dict | None


model = ChatGroq(model="openai/gpt-oss-20b", temperature=0).bind_tools(TOOLS)


def agent(state: State) -> State:
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def route_after_agent(state: State) -> str:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    if not tool_calls:
        return END
    return "check_repeat"


def check_repeat(state: State):
    """Catch the agent calling the exact same tool with the exact same args it just
    called, back to back -- the classic stuck-loop failure mode (common with
    temperature=0: nothing in the next turn differs, so it makes the same decision
    again). Instead of letting that run forever (or until the recursion limit kills
    the whole run), nudge the model to stop and act instead."""
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    current_call = {"name": tool_call["name"], "args": tool_call["args"]}

    if current_call == state.get("last_tool_call"):
        return {
            "last_tool_call": None,
            "messages": [
                HumanMessage(
                    "[system note] You just made this exact same tool call and already "
                    "have its result above -- calling it again will not give you new "
                    "information. Either act on the result you already have (e.g. call "
                    "update_section), or respond in plain text asking the user a "
                    "clarifying question. Do not repeat that tool call."
                )
            ],
        }

    return {"last_tool_call": current_call}


def route_after_check(state: State) -> str:
    last_message = state["messages"][-1]
    if last_message.type == "human":
        # check_repeat injected a correction instead of letting the call through
        return "agent"
    tool_call = last_message.tool_calls[0]
    # update_section is destructive -- gate it behind human approval.
    # read_section is read-only -- let it run straight away.
    return "approval" if tool_call["name"] == "update_section" else "tools"


def approval(state: State):
    last_message = state["messages"][-1]
    tool_call = next(tc for tc in last_message.tool_calls if tc["name"] == "update_section")

    decision = interrupt(
        {
            "tool": tool_call["name"],
            "args": tool_call["args"],
        }
    )

    if decision.get("approved"):
        return {"approved": True}

    feedback = decision.get("feedback") or "The user rejected this edit. Ask what they'd like instead."
    return {
        "approved": False,
        "messages": [HumanMessage(f"[Edit rejected by user] {feedback}")],
    }


def route_after_approval(state: State) -> str:
    return "tools" if state.get("approved") else "agent"


graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_node("check_repeat", check_repeat)
graph.add_node("approval", approval)
graph.add_node("tools", ToolNode(TOOLS))

graph.add_edge(START, "agent")
# explicit path maps so the graph visualizer can draw every branch, not just the
# ones it happens to infer -- behavior is identical either way at runtime.
graph.add_conditional_edges("agent", route_after_agent, {"check_repeat": "check_repeat", END: END})
graph.add_conditional_edges("check_repeat", route_after_check, {"agent": "agent", "approval": "approval", "tools": "tools"})
graph.add_conditional_edges("approval", route_after_approval, {"tools": "tools", "agent": "agent"})
graph.add_edge("tools", "agent")


def build_app(checkpointer):
    return graph.compile(checkpointer=checkpointer)


SYSTEM_PROMPT = (
    "You edit a markdown notes file made of ## sections: TODO, Ideas, Log. "
    "update_section always REPLACES the entire section's content, it cannot add or "
    "remove a single line on its own. So whenever the user asks to add an item, remove "
    "an item, or otherwise partially change a section, you MUST call read_section first "
    "to see what's currently there, then call update_section with the FULL new content "
    "(existing items plus your change, minus anything removed). "
    "Only skip read_section if the user already gave you the exact full content the "
    "section should end up with. Never guess or invent section content. "
    "Then confirm briefly what you changed."
)


def ask_approval(payload: dict) -> dict:
    print(f"\n[approval needed] proposed call: {payload['tool']}(args={payload['args']})")
    answer = input("Approve this edit? (y / n / n:<feedback for the agent>): ").strip()
    if answer.lower() in {"y", "yes"}:
        return {"approved": True}
    if answer.lower().startswith("n:"):
        return {"approved": False, "feedback": answer[2:].strip()}
    return {"approved": False}


RECURSION_LIMIT = 16


def run_graph(app, update, config: dict):
    """Invoke the graph, handling any number of interrupts along the way, and
    bailing out cleanly if the agent gets stuck in an unproductive loop instead
    of silently burning steps until LangGraph's recursion limit kills the run."""
    run_config = {**config, "recursion_limit": RECURSION_LIMIT}
    try:
        result = app.invoke(update, config=run_config)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            decision = ask_approval(payload)
            result = app.invoke(Command(resume=decision), config=run_config)
        return result
    except GraphRecursionError:
        print(
            f"\n[the agent didn't converge within {RECURSION_LIMIT} steps -- "
            "it may be stuck looping. Try rephrasing your request.]\n"
        )
        return app.get_state(config).values


def print_trace(result):
    print("\n--- full conversation state for this thread ---")
    for m in result["messages"]:
        tool_calls = getattr(m, "tool_calls", None)
        print(f"[{m.type}] {m.content!r}" + (f" tool_calls={tool_calls}" if tool_calls else ""))
    print("--- end state ---\n")
    print(f"Agent: {result['messages'][-1].content}\n")


def main(checkpointer):
    print("Section Updater. Sections: TODO, Ideas, Log. Type 'quit' to exit.\n")

    app = build_app(checkpointer)

    thread_id = input("Thread id (reuse the same id to continue a conversation): ").strip() or "default"
    config = {"configurable": {"thread_id": thread_id}}

    existing = app.get_state(config)
    is_new_thread = not existing.values.get("messages")
    if is_new_thread:
        print(f"[starting new thread '{thread_id}']\n")
    else:
        print(f"[resuming thread '{thread_id}', {len(existing.values['messages'])} messages already stored]\n")

    # If a previous run left this thread paused on an approval, settle it first.
    if existing.next:
        print(f"[thread '{thread_id}' was paused awaiting approval -- resuming it now]")
        pending_payload = existing.tasks[0].interrupts[0].value
        decision = ask_approval(pending_payload)
        result = run_graph(app, Command(resume=decision), config)
        print_trace(result)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break

        new_messages = [HumanMessage(user_input)]
        if is_new_thread:
            new_messages = [SystemMessage(SYSTEM_PROMPT)] + new_messages
            is_new_thread = False

        # Only the new message(s) go in -- the checkpointer loads this thread's
        # prior history from disk and merges it in via the add_messages reducer.
        # last_tool_call is reset here so check_repeat only guards within THIS
        # turn's tool-call loop, not against a legitimate repeat in a later turn.
        result = run_graph(app, {"messages": new_messages, "last_tool_call": None}, config)

        print_trace(result)


if __name__ == "__main__":
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
        main(checkpointer)
