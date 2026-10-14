---
name: sap-btp-ai-capabilities
description: This skill serves as context of the general AI Capabilities available at SAP BTP for producing PoC.
---

# SAP BTP AI Capabilities (PoC Context)

Use this skill when the conversation needs general SAP BTP AI capability context to shape or validate a PoC direction.

## Capability Matrix

| Category | Supported Project Type | What it does |
| --- | --- | --- |
| Information Analysis & Processing | Summarization of Text | Summarize long text (docs, emails, logs) into key points and actions. |
| Information Analysis & Processing | Translation of Text | Translate text between languages, optionally keeping terminology consistent. |
| Information Analysis & Processing | Sentiment Analysis of Text | Detect sentiment (positive/neutral/negative) and urgency signals in text. |
| Information Analysis & Processing | Question & Answers on Enterprise Knowledge Base | Ask questions over enterprise documents with grounded answers (often with citations). |
| Information Analysis & Processing | Information Classification in Categories | Classify text/documents into configured categories (routing, tagging, compliance buckets). |
| Information Analysis & Processing | Information Extraction from Documents | Extract structured fields/line items from PDFs/images/spreadsheets into JSON/Excel. |
| Information Analysis & Processing | Image Analysis | Analyze an image to detect elements/structure and produce a structured interpretation. |
| Content Creation | Generate Text based on Generic Knowledge | Create explanations, summaries, drafts that don’t need proprietary grounding. |
| Content Creation | Generate Text based on Proprietary, Specific Knowledge | Generate text grounded in customer/proprietary docs (RAG, doc grounding). |
| Content Creation | Review and Refine Text | Improve a draft (tone, clarity, completeness), optionally with style rules. |
| Content Creation | Generate Image from Text | Generate images from prompts (diagrams, mockups, marketing visuals). |
| Content Creation | Describe Images | Turn images into text descriptions, summaries, or extracted meaning. |
| Conversational Interaction | Conversational Interaction with Applications | Chat UI that triggers actions in workflows (create/update/check status) via tools/APIs. |
| Conversational Interaction | Conversational Interaction with Analytics | Chat over governed metrics/dashboards, returning tables and insights. |

## Usage Guidance

- Use this matrix as a capability shortlist when framing PoC options.
- Map each use-case idea to one or more supported project types before selecting implementation patterns.
- Keep recommendations aligned with available enterprise data, systems, and governance constraints.
