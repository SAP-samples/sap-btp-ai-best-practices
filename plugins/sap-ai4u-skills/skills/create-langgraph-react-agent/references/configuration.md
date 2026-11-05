# Configuration reference

## Model suppliers

- `openai`: SAP `ChatOpenAI` subclass with `proxy_model_name`, Responses API
  enabled, and the SDK 7.2.0 chat-only `n` default removed from Responses calls.
- `gemini`: SAP `ChatGoogleGenerativeAI` with `proxy_model_name`.
- `claude`: SAP `ChatBedrockConverse` with its deployment `model_name` and
  derived non-empty Bedrock `model` ID.

Keep deployment names in YAML. Verify the selected SAP AI Core resource group
instead of guessing a fallback model. Images and PDFs enter through the common
`Attachment` model, which applies Claude's stricter PDF document-name rules.
Structured output is an optional second model call after the tool loop.

## Dynamic skills

The configured skill root contains one directory per skill. Folder name and
frontmatter `name` must match. `load_skill` accepts several names, removes
duplicates without changing order, and fails the whole call when any name is
invalid. It concatenates `SKILL.md` first, then recursive UTF-8 text files in
lexical order. Binary files are listed but not injected; symlinks are rejected.

## MCP

Configure `stdio`, `sse`, or `streamable_http` servers below `mcp.servers`.
Keep secrets as `${ENV_VAR}` placeholders. Enabled required servers fail agent
startup when unavailable; optional servers report degraded diagnostics. MCP
tool names are prefixed with the server name.

## HANA memory

Set `memory.enabled: true` to use the standard `HANA_ADDRESS`, `HANA_PORT`,
`HANA_USER`, `HANA_PASSWORD`, and `HANA_ENCRYPT` values. Use the runtime user's
current schema. The runtime automatically creates, migrates, or validates its
table and persists the full message history (tool calls and results included,
attachment bytes excluded).

```yaml
memory:
  enabled: true
  table_name: LANGGRAPH_AGENT_MEMORY
  max_messages: 200        # hard ceiling; trimming restarts on a user turn
  lease_seconds: 300       # must exceed the longest turn
  lock_wait_seconds: 30    # then ContextBusyError
summarization:
  enabled: true
  trigger_tokens: 200000   # disable with native PDF attachments (see below)
  keep_messages: 6
  # model: {provider: openai, name: <cheaper deployment>}
tool_output:
  max_output_characters: 30000
approvals:
  tools: []                # tool names that need the user's "yes" first
```

Turns on one context are serialized across instances by a HANA lease and a
versioned save; do not replace them with an in-process lock. Approval-gated
tools require memory. Details: `assets/template/docs/mcp-and-hana.md`.

## A2A server

The `a2a` block controls a thin ASGI adapter around the same `AgentRuntime` used
by `ask` and `chat`:

```yaml
a2a:
  enabled: true
  name: Portable LangGraph Agent
  description: Skill-aware LangGraph agent with MCP tools
  version: 0.1.0
  public_url: ${A2A_PUBLIC_URL}
  host: 127.0.0.1
  port: 8080
```

`public_url` must be an absolute HTTP(S) base URL. Use
`http://localhost:8080` for local testing and the application route when a
human later deploys it. The server advertises A2A 1.0 and 0.3 JSON-RPC on the
same URL, serves both standard and legacy discovery paths, and maps A2A
`contextId` directly to the runtime conversation key.

The default listener is loopback-only and has no authentication middleware.
Before changing it to `0.0.0.0`, add authenticated ingress, caller-scoped A2A
server contexts, and authorization or namespacing for client-provided context
IDs. Enable HANA memory when reusing `contextId` must preserve prior model turns;
without it, the value provides correlation but not conversation history.

The first transport boundary accepts non-empty text parts. Local image/PDF
attachments remain supported through `AgentRuntime.ainvoke` and the CLI; add an
explicit A2A file policy before accepting remote file or URL parts.

## Embedding in an existing application

Validated in a FastAPI backend that runs many agent jobs concurrently:

- **Lifecycle.** Create one `AgentRuntime` per process, lazily under an
  `asyncio.Lock`, and call `aclose()` from the app's lifespan shutdown. Runs
  with distinct `context_id`s share the compiled graph, model client, and tools
  and execute in parallel; the same `context_id` is serialized.

  ```python
  _runtime: AgentRuntime | None = None
  _lock = asyncio.Lock()

  async def get_runtime() -> AgentRuntime:
      global _runtime
      async with _lock:
          if _runtime is None:
              _runtime = await AgentRuntime.create(CONFIG, extra_tools=TOOLS)
      return _runtime
  ```

- **No A2A.** Set `a2a.enabled: false`, omit `a2a_server.py` and `__main__.py`,
  and remove the `create_a2a_app` import from the package `__init__`; otherwise
  importing the runtime still requires `a2a-sdk`.
- **Native PDFs disable summarization.** `SummarizationMiddleware` estimates
  tokens with `count_tokens_approximately`, which counts base64 `file` blocks as
  text (about 100k tokens for a 300 KB PDF). Two or three documents cross the
  trigger and the first message, which holds the documents, is summarised away
  mid-run. Set `summarization.enabled: false` for attachment-heavy single-turn
  runs, or pass a `token_counter` that skips file and image blocks.
- **Run-scoped tool arguments.** A parameter typed
  `context_id: Annotated[str, InjectedState("context_id")]` (from
  `langgraph.prebuilt`) receives the run's `context_id` from `AgentState` and is
  hidden from the model's tool schema. Use the context id as the business key
  (for example an order id) and scope tool writes by it, so the model cannot
  read or forge another record's key.
- **Token usage without editing `runtime.py`.** After `ainvoke`, iterate the
  `AIMessage`s in `result.messages`; each is one model call. Emit one usage
  event per message (the `token-logger` skill's `extract_token_usage`) and sum
  input, cached input, and output tokens for a per-run total. With HANA memory
  enabled, `result.messages` also contains earlier turns; count only the
  `AIMessage`s after the last `HumanMessage`.
- **Compact tool schemas.** LangChain inlines nested Pydantic models in tool
  schemas, once per referencing field. A 28-field submission model with one
  shared value sub-model produced a 36 KB schema; a `dict[str, FieldValue]`
  field with the allowed keys listed in its `Field(description=...)` cut it to
  about 4 KB. Validate the keys inside the tool.

## Use-case extensions

Prefer small LangChain tools passed through `extra_tools`. When serving those
tools over A2A, pass them through the `runtime_factory` shown in
`assets/template/docs/a2a.md`; the built-in `serve` command creates only the
base configured runtime. Reuse the bundled A2A server when a remote agent is
required; do not add another web server,
vector database, full checkpointer, custom MCP converter, or other
infrastructure unless the use case actually needs it. Document every added
feature under the generated project's `docs/` folder.
