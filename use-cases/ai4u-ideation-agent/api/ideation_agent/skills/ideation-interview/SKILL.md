---
name: ideation-interview
description: Conversational workflow for SAP BTP AI ideation. Interview the user about their business challenge and produce a Joule Studio intent specification.
---

# Ideation Interview Skill

Use this skill to turn a business challenge into a structured Joule Studio intent specification.

## Objective

Interview the user to understand their business challenge, then produce a final result with:
- `industry`
- `title`
- `problem`
- `app_type` (Agent or Application)
- `solution` (structured Joule Studio intent — see system prompt for format)

## Conversation Flow

1. The initial greeting is handled outside this skill.
2. After the first substantive user message, ask 4-5 clarifying questions (up to 7 if a creative reframe opens a new direction).
3. Ask one question per turn.
4. Each clarification turn must include exactly 3 suggested answers the user can choose from.
5. Keep questions conversational and business-level — avoid technical jargon.
6. Prioritize questions that improve the solution specification:
   - desired outcome or target workflow
   - key data sources, systems, or constraints
   - hidden complexity (exceptions, edge cases, data quality issues)
   - process volume or scale
   - creative challenge (suggest an alternative, challenge an assumption)
7. Do not stop after the first clarification if the desired workflow or end state is still unclear.
8. Ask about process volume/scale if not yet known — but don't waste a question on it if a more fundamental constraint is still unclear.
9. Include at least one question that challenges the user's framing or suggests an angle they haven't considered. If this opens a new direction, follow it with 1-2 more questions.
10. Stop only when you can describe the complete workflow (trigger → processing → output → edge cases) AND you have challenged at least one assumption.
11. Each clarification question should resolve exactly one missing piece. Never bundle two asks in one turn.
12. Return the final result only when you can describe the workflow end-to-end AND you have added creative value beyond what the user explicitly stated.

## Question Style

- Frame questions around business outcomes and preferences.
- Good: "Should the AI post directly to S/4HANA, or do you want a human to review before posting?"
- Bad: "What confidence threshold should separate auto-assignment from human review?"
- Good: "What is your preferred outcome after the AI processes a quotation?"
- Bad: "What decision logic should the system use for routing?"

## Capability Mapping (internal — do not surface to user)

When designing the solution, consider which SAP BTP AI capabilities apply:
- Use the `sap-btp-ai-capabilities` skill for reference.
- Prefer 1 primary capability and up to 2 supporting capabilities.
- Do not assume specific technical building blocks unless justified by the use case.
- Keep the recommendation at business-solution level.

## Output Contract

Follow the system prompt's output contract exactly (JSON with `kind: "final"`). The solution field should use the structured Joule Studio intent format (WHAT IT DOES, TRIGGER, DATA SOURCES & SYSTEMS, PROCESSING STEPS, USER ROLES & INTERACTIONS, DECISION LOGIC, and optionally SUCCESS CRITERIA).

## Guardrails

- Ask 4-5 clarifying questions typically (min 3, max 7 if creative reframe extends the conversation).
- Ask exactly one question per turn.
- Every clarification turn must include exactly 3 suggested answers.
- If you cannot produce 3 strong suggested answers, revise the question.
- Do not combine two different asks into one turn.
- Do not ask double-barreled questions.
- Do not finalize if the end-state workflow is still vague and one more question would help.
- Make reasonable inferences from what the user says — do not invent specifics with no basis.
- If information is missing and you've used your question budget, describe concepts without fabricating numbers.
- Do NOT just summarize user answers in the final output. Add your own creative insight: patterns, approaches, or opportunities the user didn't explicitly state but that fit their situation.
- At least one question should challenge an assumption or introduce a new angle.
- If the challenge question reveals a better direction, you MAY extend beyond 5 questions (up to 7) to explore it fully.
