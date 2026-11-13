# AI4U agent Responses replay update

The marketplace agent template now preserves encrypted reasoning and strips
provider output item IDs from outgoing Responses history. It requests opaque
reasoning content and uses `store=False`, with `langchain-openai==1.3.5` explicitly
pinned. Function `call_id` links, tool inputs, original graph messages, and optional
HANA conversation storage remain intact.

The production application had a copy of the provider module that did not contain
the marketplace's November reasoning-discard workaround. The current marketplace
already tried to address the 404 by discarding reasoning. It therefore cannot be
identified as the direct source of that application's failure from code comparison
alone. Its broad explanation was outdated: live synthetic testing established
that encrypted reasoning replay succeeds after item IDs are removed. This update
carries that narrower fix into new scaffolds and guidance for existing projects.

## Scope

- `plugins/sap-ai4u-skills/skills/create-langgraph-react-agent/`: provider code,
  dependencies, actual-serializer tests, reasoning/tool smoke checks, configuration
  reference, core docs and the new `assets/template/docs/responses-replay.md`.
- `plugins/sap-ai4u-skills/skills/access-to-generative-ai-models/SKILL.md`: native
  self-contained replay helper and follow-up; reuse the compatible LangChain
  factory rather than unconditionally using provider response storage.
- `plugins/sap-ai4u-skills/skills/langgraph-genai-hub-setup/SKILL.md`: GPT Responses
  compatibility guidance alongside legacy Chat Completions examples.
- `plugins/sap-ai4u-skills/skills/sap-btp-ai/SKILL.md`: route ReAct work to the
  template and require multi-step smoke coverage for GPT Responses.
- `plugins/sap-ai4u-skills/CHANGELOG.md`: add the update under Unreleased, preserving
  released version metadata and earlier historical observations.

Inputs are configured SAP proxy models, prompt/history messages and local tool
results. Outputs are complete provider requests without storage-dependent item
IDs, retained encrypted reasoning, and final-text public diagnostics. HTTP 404
missing resources and HTTP 400 `invalid_encrypted_content` verification failures
are documented separately. The template does not automatically retry the latter.

```python
from template_agent.config import ModelSettings
from template_agent.providers import create_chat_model

model = create_chat_model(ModelSettings(
    provider="openai", name="YOUR_DEPLOYED_MODEL", use_responses_api=True,
    reasoning_effort="high",
))
```

Existing copied applications need a targeted provider update; updating marketplace
skills does not rewrite their code. Retain use-case callbacks, graph behavior and
existing narrow verification recovery. No HANA migrations or Cloud Foundry
deployment are part of this update.

## Checks

From a generated template project:

```bash
python -m pytest -q
ruff check template_agent/providers.py tests/test_providers.py scripts/live_model_smoke.py
python -m scripts.live_model_smoke --model openai=YOUR_DEPLOYED_MODEL
```

From the marketplace root:

```bash
python scripts/validate_package.py
```

The real-serializer tests cover parallel call/output links, preserved opaque
content and graph IDs, sync/async reasoning retention, missing opaque content,
factory behavior, unchanged Chat Completions transport and Claude model mapping.
The live checks use synthetic policy text and local arithmetic; no persisted
profile, dataset or optimizer run is touched. Syntax-check the native example and
other edited Python snippets before applying the update.

Validation completed on 13 November 2026: all 60 template tests passed, the edited
Python files passed Ruff, all 23 Python snippets in the two edited model/setup
skills parsed, and live synthetic GPT-6.1 Sol reasoning replay and arithmetic-tool
round trips succeeded. The reasoning smoke observed one encrypted reasoning item
before replay. No real application data or persistent business records were used.

Marketplace package validation passed for all 38 skills.
