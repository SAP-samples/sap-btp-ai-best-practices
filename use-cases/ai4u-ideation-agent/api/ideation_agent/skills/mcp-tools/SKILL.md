---
name: mcp-tools
description: "Guidance for using the example MCP tools exposed by this template."
---

# MCP Tools Skill

This template ships with a local MCP server named `example_tools`.

## Available tools

- `example_tools_add(a, b)`
- `example_tools_multiply(a, b)`
- `example_tools_echo(text)`
- `example_tools_get_utc_time()`
- `example_tools_prepare_ideation_documents(context_id)`
- `example_tools_finalize_ideation_export(context_id, use_case_title, ideation_payload_json, final_report_markdown)`

## Usage guidance

- Use math tools for exact arithmetic instead of mental math.
- Use UTC time tool when a time-sensitive response is required.
- Use `example_tools_prepare_ideation_documents` at ideation interview start to seed session artifacts.
- Use `example_tools_finalize_ideation_export` to write validated final JSON export plus final report.
- If tool calls fail, report the failure and continue with a best-effort answer.
