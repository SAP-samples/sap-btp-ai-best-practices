---
name: sap-btp-ai-capabilities
description: Concise AI capability catalog for grounding ideation recommendations.
---

# AI Capabilities Catalog

Use this skill when the conversation needs a grounded list of AI capabilities to match a challenge, idea, or use case to a practical solution direction.

## How To Use This Skill

- Match the user's need to the smallest set of capabilities that directly solve the problem.
- Prefer core capabilities over adjacent or optional ones.
- Explain capabilities in business language, not technical jargon.
- If multiple capabilities are needed, describe how they work together in one flow.
- Always keep track of the selected capability `id` values as well as titles.
- When a final recommendation is produced, selected capabilities should be returned as `{id, title}` objects.
- The final result must preserve the canonical capability IDs and titles exactly as they appear in this catalog.

## Selection Rules

- Choose 1 primary capability first.
- Add at most 2 supporting capabilities unless the use case clearly needs more.
- Avoid overlapping capabilities unless each one has a distinct role in the solution.
- Use workflow/action capabilities only when the use case actually needs automation or tool triggering.
- If a capability is interesting but not central, present it as secondary support rather than a core fit.
- If no strong match exists, prefer the closest practical capability instead of forcing an advanced one.
- If capability selection is uncertain, prefer fewer capabilities over speculative extras.
- Do not select `3 Chat Bots with Enterprise and/or Public Data` unless a conversational interface is a real part of the user experience.
- Do not select `9 Q&A on Documents` unless end users need to actively ask questions over document content.
- For document-driven recommendation flows, prefer extraction, ranking, and workflow capabilities before conversational ones.
- For document classification or assignment use cases, prefer `8`, `20`, and `21` before adding any conversational capability.

## Capabilities

| ID | Title | What it does |
| --- | --- | --- |
| 1 | Summarize Text | AI document summarization prioritizes important sentences, phrases, or concepts based on factors such as relevance and frequency within the text. |
| 2 | Translate Text | Machine learning algorithms analyze large amounts of text data in multiple languages and learn the relationships between words and phrases. |
| 3 | Chat Bots with Enterprise and/or Public Data | Allows users to interact with AI systems through natural language to analyze enterprise data as well as public domain information to gain insights and improve customer experiences, operations, and business decision-making. |
| 4 | Causal AI | Focuses on understanding and using cause-and-effect relationships both data and unstructured textual data, going beyond just identifying correlations, to achieve deeper insights and more robust AI systems. |
| 5 | Sentiment Analysis | The process of automatically identifying and categorizing the emotional tone or sentiment expressed in a piece of text, like a review, social media post, or article. |
| 6 | Image and Video Recognition | Analyzes image or video data so AI systems can extract meaningful information such as objects, scenes, people, and actions. |
| 7 | Image Search | Allows users to make queries with photos instead of words. |
| 8 | Document & Image Analysis | Automatically identifies, understands, and extracts specific structured data from unstructured or semi-structured documents such as PDFs, images, or web pages. |
| 9 | Q&A on Documents | Extracts specific information from text documents by posing questions. |
| 10 | Streamline Content Creation | Optimizes and automates content creation tasks, from generating ideas and drafting content to editing and distributing it. |
| 11 | Generate Image | Creates a new original image based on instructions or prompts. |
| 12 | Speech to Text | Uses AI to generate speech from text, creating voices that sound convincingly human for applications such as video narration and e-learning. |
| 13 | Behavioral and Process Recommendations Based on History | Suggests likely next actions, decisions, or preferences by analyzing patterns in past behavior and historical data. |
| 14 | Cluster Data | Groups similar data points into clusters based on shared characteristics or patterns, without prior category labels. |
| 15 | Forecast Values | Uses algorithms and data analysis to predict future values such as sales, inventory, returns, or demand based on patterns and trends. |
| 16 | Anomaly Detection | Identifies data points, events, or patterns that deviate significantly from normal or expected behavior. |
| 17 | Simulate Scenarios | Uses AI to create virtual representations of real-world events or systems. |
| 18 | Predictive Weather | Analyzes large weather datasets to predict future weather conditions more accurately and efficiently. |
| 19 | Predictive Maintenance | Uses data analysis, sensors, and machine learning to monitor equipment and predict failures before they happen. |
| 20 | Rank Alternatives | Sorts or prioritizes options or results based on relevance, value, or likelihood of success. |
| 21 | Intelligently Activate Workflows/Tools | Uses AI-driven automation to trigger actions, workflows, or tools in order to streamline processes and improve decision-making. |

## Practical Matching Hints

- Text-heavy use cases often map to `Summarize Text`, `Translate Text`, `Sentiment Analysis`, or `Q&A on Documents`.
- Document automation use cases often map to `Document & Image Analysis` and `Q&A on Documents`.
- Operational prediction use cases often map to `Forecast Values`, `Anomaly Detection`, `Predictive Maintenance`, or `Predictive Weather`.
- Search, ranking, and personalization flows often map to `Image Search`, `Behavioral and Process Recommendations Based on History`, or `Rank Alternatives`.
- AI copilots and action-oriented assistants often map to `Chat Bots with Enterprise and/or Public Data` and `Intelligently Activate Workflows/Tools`.
- Visual use cases often map to `Image and Video Recognition`, `Image Search`, `Document & Image Analysis`, or `Generate Image`.

## Capability Combination Patterns

- `3 + 21`: conversational assistant plus action-taking workflow automation
- `8 + 9`: document extraction plus question answering over document content
- `8 + 21`: document analysis plus downstream business process activation
- `8 + 20`: document analysis plus ranking of the best match or best next option
- `8 + 20 + 21`: document extraction plus best-match recommendation plus workflow execution or review routing
- `3 + 20`: conversational assistant plus ranking of recommendations or alternatives
- `13 + 20`: history-based recommendations plus prioritization of next best options
- `15 + 16`: forecasting plus anomaly detection for operational planning and monitoring
- `6 + 8`: visual recognition plus document/image analysis when both scene understanding and extraction are needed

## Use With Caution

- `4 Causal AI`: do not choose this unless the use case truly depends on cause-and-effect reasoning rather than prediction or correlation.
- `17 Simulate Scenarios`: do not choose this unless the solution needs scenario modeling, virtual testing, or synthetic environment exploration.
- `12 Speech to Text`: despite the title, the current description is voice-generation oriented; only use it when spoken or audio-based interaction is genuinely part of the use case.

## Recommendation Guidance

- Recommend only the capabilities that materially improve the user's outcome.
- If one capability is the clear fit, say so directly instead of presenting a long list.
- If a capability is secondary, present it as optional support rather than a core recommendation.
- Keep the final recommendation practical, business-oriented, and implementation-aware.
- For final outputs, preserve both the selected capability IDs and titles exactly as they appear in this catalog.
- Never invent capability IDs or paraphrase capability titles in the final structured output.
- In the final recommendation, explain the operating model first and keep low-level technical implementation choices optional unless the user asks for them.
