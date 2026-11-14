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
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

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


TOOLS = [update_section]


class State(TypedDict):
    messages: Annotated[list, add_messages]


model = ChatGroq(model="openai/gpt-oss-20b", temperature=0).bind_tools(TOOLS)


def agent(state: State) -> State:
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def route_after_agent(state: State) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(TOOLS))

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", route_after_agent)
graph.add_edge("tools", "agent")

app = graph.compile()


SYSTEM_PROMPT = (
    "You edit a markdown notes file made of ## sections: TODO, Ideas, Log. "
    "When the user asks to add, change, or replace something in a section, "
    "call update_section with the FULL new content for that section "
    "(if the user is adding to existing content, include the existing items too). "
    "Then confirm briefly what you changed."
)


def main():
    print("Section Updater. Sections: TODO, Ideas, Log. Type 'quit' to exit.\n")

    history: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT)]

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break

        history.append(HumanMessage(user_input))

        result = app.invoke({"messages": history})
        history = result["messages"]  # carry this turn's messages into the next one

        print("\n--- message trace ---")
        for m in result["messages"]:
            tool_calls = getattr(m, "tool_calls", None)
            print(f"[{m.type}] {m.content!r}" + (f" tool_calls={tool_calls}" if tool_calls else ""))
        print("--- end trace ---\n")

        print(f"Agent: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
