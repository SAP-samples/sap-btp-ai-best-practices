# LangGraph ReAct Agents

This section contains three progressive Jupyter notebooks that demonstrate how to build **ReAct agents using LangGraph** on SAP BTP. The examples start with a basic tool-calling agent and build toward an enterprise procurement workflow, showing how the same underlying pattern scales to real-world business scenarios.

## Purpose

While the `native-react/` example shows the ReAct loop from scratch, these notebooks demonstrate the **framework approach**: using LangGraph's state graphs, LangChain's structured tool calling, and prebuilt components like `ToolNode` and `tools_condition` to build agents with less boilerplate and more reliability.

## Files

| File | Description |
|------|-------------|
| `01_react_agent.ipynb` | Basic ReAct agent with simple tools |
| `02_multi_step_workflow.ipynb` | Multi-agent routing with conditional edges and specialized agents |
| `03_procurement_workflow.ipynb` | Enterprise procurement workflow with business logic and mock SAP data |
| `tools.py` | Simple tools shared across notebooks 1 and 2 (`add`, `multiply`, `get_weather`) |
| `procurement_tools.py` | Enterprise tools for notebook 3 (product lookup, inventory, budget, suppliers, purchase orders) |
| `data/` | CSV files with mock enterprise data (products, inventory, budgets, suppliers) |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for required SAP AI Core environment variables |

## Notebook Progression

The three notebooks form a learning path where complexity increases through **tool design, state management, and prompt sophistication** -- not just graph topology.

### Notebook 1: Basic ReAct Agent

**Goal**: Introduce the LangGraph ReAct pattern with the simplest possible graph.

The graph has two nodes:
- **assistant**: calls the LLM with tools bound via `bind_tools()`
- **tools**: a `ToolNode` that executes whatever tools the LLM requested

Routing uses `tools_condition`: if the LLM response contains tool calls, go to the tools node; otherwise, end.

```
START --> assistant --> [tools_condition] --> tools --> assistant --> ... --> END
                                         \-> END
```

**Demonstrates**:
- SAP GenAI Hub initialization with the LangChain `ChatOpenAI` proxy
- LangChain `@tool` decorator for automatic JSON schema generation
- `MessagesState` for message history management
- Single-step and multi-step tool calling
- Parallel tool execution (the LLM can request multiple tools in one response)

### Notebook 2: Multi-Step Workflow with Conditional Routing

**Goal**: Show how to route queries to **specialized agents** based on classification.

The graph adds an LLM-based classifier node at the entry point. It determines the query category (math, weather, or general) and routes to the appropriate agent. Each agent has access only to its relevant tools:

- **math_agent**: `add` and `multiply` only
- **weather_agent**: `get_weather` only
- **general_agent**: all tools

```
                    START
                      |
                 [classifier]
                      |
              route_by_category
               /      |      \
       math_agent  weather_agent  general_agent
              \      |      /
            check_tool_calls
              /            \
        has tool calls    no tool calls
             |                |
          [tools]            END
             |
       route_back_to_agent  (loops back to the originating agent)
```

**Demonstrates**:
- Custom state schemas extending `MessagesState` with a `category` field
- LLM-based classification as a routing mechanism
- Scoped tool access per agent (security and cost control)
- Domain-specific system prompts for each specialized agent
- Conditional edges with custom routing functions

### Notebook 3: Enterprise Procurement Workflow

**Goal**: Show that a **simple graph topology combined with sophisticated tools and prompts** can handle complex enterprise processes.

This notebook uses the same two-node graph as Notebook 1, but replaces the toy tools with six enterprise-grade tools backed by CSV mock data. The system prompt encodes a detailed 6-step procurement procedure:

1. Extract request details (product, quantity, plant, department)
2. Look up product in the catalog
3. Check inventory at the requested plant
4. Validate department budget
5. Check supplier constraints (lead time, minimum order)
6. Create a purchase order draft

```
START --> assistant --> [tools_condition] --> tools --> assistant --> ... --> END
                                         \-> END
```

**Demonstrates**:
- Enterprise tool design with contextual responses (e.g., inventory check returns stock at all plants, not just the requested one)
- Procedural workflow encoded in the system prompt
- Graceful failure handling driven by the LLM (budget exceeded, out of stock, product not found)
- Mock data loaded from CSV files simulating SAP systems (Material Master, Warehouse Management, Controlling, Vendor Master)

## Tools

### Simple Tools (`tools.py`)

Used by Notebooks 1 and 2:

| Tool | Signature | Description |
|------|-----------|-------------|
| `add` | `add(a: float, b: float) -> float` | Returns the sum of two numbers |
| `multiply` | `multiply(a: float, b: float) -> float` | Returns the product of two numbers |
| `get_weather` | `get_weather(city: str) -> str` | Returns mock weather from a hardcoded dictionary |

### Procurement Tools (`procurement_tools.py`)

Used by Notebook 3:

| Tool | Purpose | Key Behavior |
|------|---------|-------------|
| `lookup_product` | Material Master lookup | Searches product catalog by name (substring match), returns product ID, price, category, supplier |
| `check_inventory` | Warehouse Management | Returns available stock at the requested plant **plus** stock levels at all other plants for context |
| `validate_budget` | Cost Center validation | Checks if the department's remaining budget covers the purchase amount, returns APPROVED or REJECTED |
| `get_supplier_info` | Vendor Master lookup | Returns supplier lead time, minimum order quantity, and reliability rating |
| `search_alternative_products` | Catalog search | Finds cheaper alternatives in the same category (used when original product is over budget or unavailable) |
| `create_purchase_order_draft` | Purchase Order creation | Generates a formatted PO draft with calculated totals and a timestamped PO number |

### Mock Data (`data/`)

| File | Content | Records |
|------|---------|---------|
| `products.csv` | Product catalog with IDs, names, prices, categories, and supplier links | 8 products across 5 categories |
| `inventory.csv` | Stock levels per product and plant (units in stock, reserved units) | 3 plants: Berlin, Munich, Dublin |
| `budgets.csv` | Department budgets with total and remaining amounts | 4 departments |
| `suppliers.csv` | Supplier details with lead times, minimum orders, and reliability ratings | 4 suppliers |

## Common Code Pattern

All three notebooks follow the same initialization pattern:

```python
# 1. Load environment and initialize LLM
from dotenv import load_dotenv
load_dotenv()
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
llm = ChatOpenAI(proxy_model_name="gpt-4.1", temperature=0.0)

# 2. Define tools with @tool decorator
from langchain_core.tools import tool
@tool
def my_tool(arg: str) -> str:
    """Docstring becomes the tool description for the LLM."""
    return result

# 3. Build graph
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

graph = StateGraph(MessagesState)
graph.add_node("assistant", assistant_node)
graph.add_node("tools", ToolNode(tools=[...]))
graph.add_edge(START, "assistant")
graph.add_conditional_edges("assistant", tools_condition)
graph.add_edge("tools", "assistant")
agent = graph.compile()

# 4. Invoke
result = agent.invoke({"messages": [("user", "your question")]})
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

3. Open the notebooks in order:

   ```bash
   jupyter lab 01_react_agent.ipynb
   ```

## Key Takeaways

- **Framework benefits**: LangGraph handles message state, tool execution, and routing logic, letting you focus on tool design and business logic instead of parsing and loop management.
- **Structured tool calling** (JSON schemas via `bind_tools`) is more reliable than regex-based text parsing used in the native approach.
- **Graph complexity is not always necessary**: Notebook 3 shows that a simple two-node graph with well-designed tools and a detailed system prompt can handle complex multi-step business workflows.
- **Tool design matters more than graph design**: Tools that return contextual information (like inventory at all plants) reduce the number of agent iterations and improve response quality.
- **Scoped tool access** (Notebook 2) is a practical pattern for limiting cost, reducing errors, and enforcing domain boundaries in multi-agent systems.
