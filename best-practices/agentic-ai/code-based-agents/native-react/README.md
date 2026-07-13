# Native ReAct Agent

This example implements a **ReAct (Reasoning and Acting) agent from scratch** in Python, using only the SAP Generative AI Hub's native OpenAI-compatible API. No agent framework is used -- the entire loop is built manually to demonstrate how ReAct agents work at a fundamental level.

## Purpose

Before using frameworks like LangGraph, it is valuable to understand the underlying mechanics: how the LLM is prompted to produce structured output, how that output is parsed into tool calls, how observations are fed back, and how the loop terminates. This example makes all of those steps explicit.

## Files

| File | Description |
|------|-------------|
| `native_react_agent.ipynb` | Jupyter notebook with the full agent implementation, explained step by step |
| `tools.py` | Three simple tools used by the agent (`add`, `multiply`, `get_weather`) |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for required SAP AI Core environment variables |

## How It Works

The notebook walks through six steps, each building on the previous one:

### Step 1 -- Tool Registration

Tools are registered using a `ToolInfo` dataclass that pairs each function with its name, description, and parameter signature. A `tool_map` dictionary provides fast lookup by name. This mirrors how frameworks like LangChain register tools, but does so explicitly.

### Step 2 -- System Prompt Construction

A `build_system_prompt()` function dynamically generates the system prompt. It instructs the LLM to follow strict formatting rules:

- `Thought:` -- reasoning about the current state
- `Action: tool_name(args)` -- a tool invocation
- `Final Answer:` -- the conclusive response when no more tool calls are needed

The prompt includes the list of available tools and their descriptions, generated from the tool registry.

### Step 3 -- LLM Integration

A `call_llm()` function sends messages to the SAP Generative AI Hub using the native OpenAI-compatible client (`gen_ai_hub.proxy.native.openai`). The model is configured with `temperature=0.0` for deterministic output.

### Step 4 -- Output Parsing and Tool Execution

Two functions handle the LLM's response:

- **`parse_action()`** uses regex to extract the tool name and arguments from `Action: tool_name(args)` patterns.
- **`execute_tool()`** looks up the tool in the registry and calls it with the parsed arguments.
- **`get_final_answer()`** extracts the text after `Final Answer:` when the agent is done.

### Step 5 -- The Agent Loop

The `ReactAgent` class ties everything together. Its `run()` method implements the core loop:

```
[Question] --> call LLM --> has Final Answer? --> YES --> return answer
                  ^                |
                  |               NO
                  |                |
                  |                v
                  +--- parse Action, execute tool, append Observation to scratchpad
```

The **scratchpad** is the key mechanism: it accumulates the full chain of Thought/Action/Observation entries and is re-sent to the LLM each iteration, giving the model context about what it has already done.

### Step 6 -- Demonstrations

Four example queries show progressive complexity:

1. **Single tool call**: `15 * 7` -- one multiplication
2. **Multi-step reasoning**: `(12 + 8) * 3` -- addition followed by multiplication
3. **String arguments**: Weather lookup for Tokyo
4. **Combined tools**: Weather data plus arithmetic in one query

## Tools

The three tools in `tools.py` are intentionally simple so that the focus stays on the agent mechanics:

| Tool | Signature | Description |
|------|-----------|-------------|
| `add` | `add(a: float, b: float) -> float` | Returns the sum of two numbers |
| `multiply` | `multiply(a: float, b: float) -> float` | Returns the product of two numbers |
| `get_weather` | `get_weather(city: str) -> str` | Returns mock weather data from a hardcoded dictionary (Berlin, New York, Tokyo, London, Paris) |

## Architecture

```
User Question
      |
      v
+---------------------+
|   ReactAgent.run()   |
|                      |
|  system_prompt       |     +-----------+
|  scratchpad -------->|---->|  LLM Call  |
|                      |     +-----------+
|                      |           |
|  parse_action() <----|<----------+
|  execute_tool()      |
|  scratchpad += obs   |
|                      |
|  Final Answer? ----->|--> return
+---------------------+
```

## Setup

1. Copy `.env.example` to `.env` and fill in your SAP AI Core credentials:

   ```
   AICORE_AUTH_URL=https://<your-auth-url>
   AICORE_CLIENT_ID=<your-client-id>
   AICORE_CLIENT_SECRET=<your-client-secret>
   AICORE_BASE_URL=https://<your-aicore-url>
   AICORE_RESOURCE_GROUP=<your-resource-group>
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the notebook:

   ```bash
   jupyter lab native_react_agent.ipynb
   ```

## Key Takeaways

- A ReAct agent is fundamentally a **loop** around an LLM call, with structured output parsing and tool execution.
- The **system prompt** is the contract that defines how the LLM should format its responses. Prompt discipline is critical.
- The **scratchpad** provides multi-step memory by replaying the full conversation history on each iteration.
- Building from scratch reveals that agent frameworks automate these same steps -- understanding them makes framework usage more effective and debugging much easier.
