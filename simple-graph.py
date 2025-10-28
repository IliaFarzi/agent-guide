from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient, load_mcp_tools_sync  # <-- synchronous loader

from dotenv import load_dotenv

load_dotenv()

# -------------------------
# MCP server config
# -------------------------
MCP_SERVERS = {
    "jupyter": {
        "command": "jupyter-mcp-server",
        "args": [
            "start",
            "--transport", "stdio",
            "--document-url", "http://localhost:8888",
            "--runtime-url", "http://localhost:8888",
            "--document-token", "mcp",
            "--runtime-token", "mcp",
            "--document-id", "test.ipynb",
        ],
        "transport": "stdio",
    }
}

# -------------------------
# Define the LangGraph state
# -------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]

# -------------------------
# Node functions
# -------------------------
def chatbot_node(state: State):
    # last message content -> LLM response
    return {"messages": [llm.invoke(state["messages"])]}

def call_tools(state: State):
    last_message = state["messages"][-1]
    # if LLM wants to call a tool, redirect to tool node
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END

# -------------------------
# Initialize LLM
# -------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)# add     base_url="https://api.metisai.ir/openai/v1" for using metis

# -------------------------
# Load MCP tools synchronously
# -------------------------
def get_tools_sync():
    client = MultiServerMCPClient(MCP_SERVERS)
    session = client.session("jupyter")  # synchronous session
    tools = load_mcp_tools_sync(session)
    print("Available Jupyter MCP Tools:", [t.name for t in tools])
    return tools

# -------------------------
# Build the LangGraph workflow
# -------------------------
def build_graph_sync():
    tools = get_tools_sync()
    tool_node = ToolNode(tools)

    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph_builder.add_conditional_edges("chatbot", call_tools)

    return graph_builder.compile()

# -------------------------
# Stream user interaction
# -------------------------
def stream_graph(graph, user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# -------------------------
# Main entry
# -------------------------
graph = build_graph_sync()
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    stream_graph(graph, user_input)

