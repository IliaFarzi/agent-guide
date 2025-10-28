from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage

from config import OPENAI_API_KEY
from tools import search_web, get_weather

llm = ChatOpenAI(model="o4-mini", api_key=OPENAI_API_KEY)# add     base_url="https://api.metisai.ir/openai/v1" for using metis

tools = [search_web, get_weather]
llm_with_tools = llm.bind_tools(tools)

system_message = SystemMessage(
    content="You are an AI assistant with access to tools. "
            "If the user asks about current events, facts, or knowledge you may not know, "
            "always use the `search_web` tool to find reliable information. "
            "For weather-related queries, prefer the `get_weather` tool."
)

queries = [
    "What is the current weather in Trivandrum today",
    "Can you tell me about Kerala situation today",
    "Why is the sky blue?"
]

def run_query(query: str):
    print(f"\n--- Query: {query} ---")
    response = llm_with_tools.invoke([system_message, HumanMessage(content=query)])
    if isinstance(response, AIMessage) and response.tool_calls:
        tool_messages = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool:
                tool_result = tool.invoke(tool_args)
                print(f"Executed tool {tool_name} → {tool_result}")
                tool_messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                )
        final_response = llm_with_tools.invoke([system_message, response] + tool_messages)
        print("Final Answer:", final_response.content)
    else:
        print("Final Answer:", response.content)

for q in queries:
    run_query(q)
