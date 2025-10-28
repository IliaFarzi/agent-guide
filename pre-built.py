from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY
from tools import search_web, get_weather

# system prompt is used to inform the tools available to when to use each
system_prompt = """Act as a helpful assistant.
    Use the tools at your disposal to perform tasks as needed.
        - get_weather: whenever user asks get the weather of a place.
        - search_web: whenever user asks for information on current events or if you don't know the answer.
    Use the tools only if you don't know the answer.
    """

llm = ChatOpenAI(model="o4-mini", api_key=OPENAI_API_KEY)# add     base_url="https://api.metisai.ir/openai/v1" for using metis

tools = [search_web, get_weather]
llm_with_tools = llm.bind_tools(tools)
agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)

# Let’s query the agent to see the result.
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "What is the weather in Theran now")]}

print_stream(agent.stream(inputs, stream_mode="values"))