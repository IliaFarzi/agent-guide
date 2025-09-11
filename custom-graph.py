# import the required methods
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from typing import Literal

from config import OPENAI_API_KEY
from tools import search_web, get_weather

# define a tool_node with the available tools
tools = [search_web, get_weather]
tool_node = ToolNode(tools)

llm = ChatOpenAI(model="o4-mini", api_key=OPENAI_API_KEY)
llm_with_tools = llm.bind_tools(tools)

# define functions to call the LLM or the tools
def call_model(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def call_tools(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

# initialize the workflow from StateGraph
workflow = StateGraph(MessagesState)

# add a node named LLM, with call_model function. This node uses an LLM to make decisions based on the input given
workflow.add_node("LLM", call_model)

# Our workflow starts with the LLM node
workflow.add_edge(START, "LLM")

# Add a tools node
workflow.add_node("tools", tool_node)

# Add a conditional edge from LLM to call_tools function. It can go tools node or end depending on the output of the LLM. 
workflow.add_conditional_edges("LLM", call_tools)

# tools node sends the information back to the LLM
workflow.add_edge("tools", "LLM")

agent = workflow.compile()
# display(Image(agent.get_graph().draw_mermaid_png()))

for chunk in agent.stream(
    {"messages": [("user", "Will it rain in Trivandrum today?")]},
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()
