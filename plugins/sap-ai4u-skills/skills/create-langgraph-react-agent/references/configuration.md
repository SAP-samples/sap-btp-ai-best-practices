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
current schema. The runtime automatically creates or validates its project-
specific table and persists only bounded user/final-assistant JSON turns.

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

## Use-case extensions

Prefer small LangChain tools passed through `extra_tools`. When serving those
tools over A2A, pass them through the `runtime_factory` shown in
`assets/template/docs/a2a.md`; the built-in `serve` command creates only the
base configured runtime. Reuse the bundled A2A server when a remote agent is
required; do not add another web server,
vector database, full checkpointer, custom MCP converter, or other
infrastructure unless the use case actually needs it. Document every added
feature under the generated project's `docs/` folder.
