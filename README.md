# 🤖 AI Agent Development with LangGraph - Student Guide

A comprehensive educational repository for learning AI agent development using LangGraph, LangChain, and OpenAI. This project progressively teaches you to build increasingly sophisticated AI agents with tool calling, memory, monitoring, and custom workflows.

## 📚 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Learning Path](#learning-path)
- [Examples Explained](#examples-explained)
- [Key Concepts](#key-concepts)
- [API Keys](#api-keys)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

---

## 🎯 Overview

This repository demonstrates various approaches to building AI agents, from simple tool-calling assistants to complex agents with memory and monitoring capabilities. You'll learn:

- How TOOL CALLING works in LLM-based agents
- Building CUSTOM WORKFLOWS with LangGraph
- Implementing PERSISTENT MEMORY using MongoDB
- MONITORING agent performance with LangSmith
- Using PRE-BUILT AGENT patterns
- Integrating MCP (Model Context Protocol) tools

---

## 📋 Prerequisites

Before starting, ensure you have:

1. **Python 3.9+** installed
2. **Basic Python knowledge** (functions, classes, imports)
3. **Understanding of APIs** (RESTful concepts)

---

## 🚀 Installation

### Method 1: Using the Setup Script (Recommended for Linux/Mac)

```bash
chmod +x setup.sh
./setup.sh
```

### Method 2: Manual Installation (Windows/All Platforms)

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install required packages using requirements.txt
pip install -r requirements.txt

# OR install manually
pip install langchain langchain_openai langgraph langsmith tavily-python
pip install langchain_community python-dotenv langgraph-checkpoint-mongodb requests
```

---

## ⚙️ Configuration

1. **Create your `.env` file:**

   Copy the example file:
   ```bash
   cp example.env .env
   ```

2. **Edit `.env` with your API keys:**
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   WEATHER_API_KEY=your_weather_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

See the [API Keys](#api-keys) section for where to obtain these.

---

## 🎓 Learning Path

Follow these examples in order for the best learning experience:

### 1. **Basic Tool Calling** (`main.py`) ⭐ Start Here
Understanding how LLMs call tools without a graph structure.

### 2. **Pre-built Agent** (`pre-built.py`) ⭐⭐ Easy
Using LangGraph's ready-made agent patterns.

### 3. **LangGraph basic** (`simple-graph.py`) ⭐⭐ Easy
Exploring how LangGraph works

### 4. **Custom Graph Workflow** (`custom-graph.py`) ⭐⭐⭐ Intermediate
Building your own agent workflow from scratch.

### 5. **Agent with Memory** (`memory-agent.py`) ⭐⭐⭐⭐ Advanced
Adding persistent memory with MongoDB.

### 6. **Agent with Monitoring** (`monitoring-agent.py`) ⭐⭐⭐⭐ Advanced
Tracking agent performance with LangSmith.



---

## 📖 Examples Explained

### Example 1: Basic Tool Calling (`main.py`)

**What you'll learn:** How LLMs decide when and how to use tools.

**Key Concepts:**
- Binding tools to an LLM
- Tool invocation and execution
- Message types (Human, AI, Tool)

**To run:**
```bash
python main.py
```

**Code breakdown:**
```python
# 1. Create LLM with tools
llm_with_tools = llm.bind_tools(tools)

# 2. LLM decides to use tools or answer directly
response = llm_with_tools.invoke([system_message, HumanMessage(content=query)])

# 3. Execute tools if requested
if response.tool_calls:
    # Execute each tool
    for tool_call in response.tool_calls:
        tool_result = tool.invoke(tool_args)

# 4. Get final answer from LLM
final_response = llm_with_tools.invoke([system_message, response] + tool_messages)
```

---

### Example 2: Pre-built Agent (`pre-built.py`)

**What you'll learn:** Using LangGraph's pre-built agent patterns for rapid development.

**Key Concepts:**
- `create_react_agent()` function
- React pattern (Reasoning + Acting)
- Streaming responses

**To run:**
```bash
python pre-built.py
```

**Why use pre-built agents?**
- ✅ Faster development
- ✅ Proven patterns
- ✅ Built-in best practices
- ❌ Less customization

---

### Example 3: Custom Graph Workflow (`custom-graph.py`)

**What you'll learn:** Creating your own agent architecture with full control.

**Key Concepts:**
- StateGraph and nodes
- MessageState management
- Conditional edges
- Graph compilation

**To run:**
```bash
python custom-graph.py
```

**Workflow visualization:**
```
START → LLM → {tool_calls?} → tools → LLM → END
                   ↓ no
                  END
```

**Node Functions:**
- `call_model()`: Invokes LLM with current state
- `call_tools()`: Decides whether to use tools or end
- `tool_node`: Executes tools (using `ToolNode`)

---

### Example 4: Agent with Memory (`memory-agent.py`)

**What you'll learn:** Adding persistent memory to maintain conversation context.

**Key Concepts:**
- MongoDB as a checkpoint store
- Thread management
- State persistence
- LangSmith tracing

**Prerequisites:**
- MongoDB running on `localhost:27017`
- Install MongoDB: https://www.mongodb.com/try/download/community

**To run:**
```bash
cheong 1: Start MongoDB
mongod

# Terminal 2: Run the agent
python memory-agent.py
```

**Config object:**
```python
config = {
    "configurable": {
        "thread_id": "weather1"  # Unique conversation thread
    }
}
```

**Benefits of memory:**
- Remember previous messages
- Context across sessions
- Multi-turn conversations
- Resume interrupted sessions

---

### Example 5: Agent with Monitoring (`monitoring-agent.py`)

**What you'll learn:** Tracking and debugging agent behavior with LangSmith.

**Key Concepts:**
- LangSmith tracing
- `@traceable` decorator
- Performance monitoring
- Debug logging

**Prerequisites:**
- LangSmith account (free tier available)
- Set environment variables for LangSmith

**To run:**
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_langsmith_api_key
python monitoring-agent.py
```

**What you can monitor:**
- Tool execution time
- LLM token usage
- Success/failure rates
- Cost tracking
- Debug traces

---
## 🔑 Key Concepts

### What is an AI Agent?

An AI agent is an autonomous system that:
1. **Perceives** its environment (user input)
2. **Decides** what action to take (using LLM reasoning)
3. **Acts** using available tools
4. **Learns** from outcomes (optional)

### Core Components

#### 1. **State Management**
- Store conversation history
- Track current context
- Manage tool inputs/outputs

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

#### 2. **Nodes**
- Individual processing units
- Can be LLM calls, tool executions, or custom logic
- Return updated state

```python
def call_model(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

#### 3. **Edges**
- Connect nodes together
- Can be:
  - **Regular edges**: Always follow this path
  - **Conditional edges**: Decision-making based on state

```python
workflow.add_edge(START, "LLM")  # Regular edge
workflow.add_conditional_edges("LLM", call_tools)  # Conditional edge
```

#### 4. **Tools**
- External capabilities the agent can use
- Must be decorated with `@tool`
- Have a name, description, and function

```python
@tool
def get_weather(query: str):
    """Get current weather for a location"""
    return weather_data
```

---

## 🔐 API Keys

### OpenAI API Key
**Where to get:** https://platform.openai.com/api-keys
- Create an account
- Generate a new API key
- Add credits to your account

**Alternative (Metis AI):**
```python
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    base_url="https://api.metisai.ir/openai/v1"  # Iranian users
)
```

### Weather API Key
**Where to get:** https://www.weatherapi.com/
- Sign up for free account
- Copy your API key from dashboard
- Free tier: 1 million calls/month

### Tavily API Key
**Where to get:** https://tavily.com/
- Sign up for free account
- Get your API key
- Free tier available for development

---

## 🛠️ Troubleshooting

### Problem: "Please set OPENAI_API_KEY in your .env file"

**Solution:**
1. Check that `.env` file exists in the project root
2. Verify the key is spelled correctly: `OPENAI_API_KEY=...`
3. Restart your terminal/IDE after creating `.env`

### Problem: "ModuleNotFoundError: No module named 'langchain'"

**Solution:**
```bash
# Install all required packages
pip install -r requirements.txt
```

### Problem: MongoDB connection error in `memory-agent.py`

**Solution:**
1. Install MongoDB: https://www.mongodb.com/try/download/community
2. Start MongoDB service:
   - Windows: Start MongoDB service from Services
   - Mac: `brew services start mongodb-community`
   - Linux: `sudo systemctl start mongod`
3. Verify it's running: `mongosh` (or check port 27017)

### Problem: Agent not using tools

**Solutions:**
1. Check tool descriptions are clear and detailed
2. Ensure system prompt mentions when to use tools
3. Verify tools are properly bound: `llm.bind_tools(tools)`
4. Check if tool calls exist: `response.tool_calls`

### Problem: Memory not persisting

**Solutions:**
1. Use consistent `thread_id` across runs
2. Ensure MongoDB is running
3. Check MongoDB connection string
4. Verify checkpointer is passed to `.compile()`

---

## 🎯 Practice Exercises

Try these to reinforce your learning:

### Exercise 1: Add a New Tool
Create a `calculate` tool that performs basic math operations and add it to `main.py`.

### Exercise 2: Modify System Prompt
Change the system prompt in `pre-built.py` to make the agent more conversational.

### Exercise 3: Add a Node
In `custom-graph.py`, add a "validator" node that checks tool outputs before sending to LLM.

### Exercise 4: Implement Custom Memory
Replace MongoDB with file-based checkpointing in `memory-agent.py`.

### Exercise 5: Create Your Own Agent
Build an agent for a specific domain (e.g., coding assistant, research helper) using what you learned.


---

## 🤝 Contributing

Found a bug or have a suggestion? Feel free to:
1. Open an issue describing the problem
2. Submit a pull request with improvements
3. Share your agent creations!

---

## 📝 License

This educational repository is free to use for learning purposes.

---

## 🙏 Acknowledgments

- Built with [LangGraph](https://langchain-ai.github.io/langgraph/) and [LangChain](https://python.langchain.com/)
- Powered by OpenAI, Tavily, and WeatherAPI
- Designed for educational purposes

---

**Happy Learning! 🚀**

If you have questions or get stuck, review the code comments and error messages carefully. Most issues stem from configuration problems rather than code errors.

