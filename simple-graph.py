from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools

from config import OPENAI_API_KEY


# -------------------------
# Define the state
# -------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]

# -------------------------
# Initialize OpenAI LLM
# -------------------------
llm = ChatOpenAI(model="o4-mini", api_key=OPENAI_API_KEY)

# -------------------------
# Load Wikipedia and LLM-Math tools
# -------------------------
tools = load_tools(["wikipedia"], llm=llm)


# -------------------------
# Create ToolNode
# -------------------------
tool_node = ToolNode(tools)

# -------------------------
# Node functions
# -------------------------
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def call_tools(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END

# -------------------------
# Build the LangGraph workflow
# -------------------------
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)
graph_builder.add_conditional_edges("chatbot", call_tools)

# Compile the graph
graph = graph_builder.compile()

# -------------------------
# Stream interaction
# -------------------------
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# -------------------------
# Run loop
# -------------------------
if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
