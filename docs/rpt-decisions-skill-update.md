# Direct RPT decisions skill update

The marketplace skill now targets the verified `sap-rpt-decisions` AI Core deployment and `/systemone` contract. It retires the local tabular translator, replaces its assets/tests with the direct client and offline contract checks from `rpt-forecast-decisions`, and bundles synthetic examples extracted from the fully executed notebook on that branch. No credentials, real customer dataset, or notebook execution history is copied.

Resources under `plugins/sap-ai4u-skills/skills/rpt-decisions/` include `SKILL.md`, `docs/rpt-decisions.md`, `assets/rpt_decisions_client.py`, `assets/decision_examples.json`, `assets/requirements.txt`, `assets/.env.example`, `tests/test_rpt_decisions.py`, and `scripts/live_test_rpt_decisions.py`. The shared client remains at the sibling `rpt-1/assets/rpt_client_api.py`; it matches the verified source branch and is reused unchanged.

Inputs are a state, typed questions and labeled examples; outputs are raw endpoint responses and optional diagnostic summaries. The guide documents probability versus confidence, expected score versus selected level, server limits, retries, policy design, structured/list inputs, held-out evaluation, HANA persistence guidance, and manual deployment boundaries.

```python
from rpt_decisions_client import RPTDecisionsClientAPI
result = RPTDecisionsClientAPI().decide(request)
confidence = result["answers"]["priority"]["confidence"]
```

Validation from the skill folder:

```bash
python tests/test_rpt_decisions.py
python -I tests/test_rpt_decisions.py
python scripts/live_test_rpt_decisions.py --env-file /path/to/.env --output /tmp/decisions.json
```

Offline checks require no credentials. The optional live command sends eight synthetic requests to an existing deployment and reports model quality without treating it as a deterministic assertion. This update uses results already measured on the source branch; live execution is reported separately if performed.

Validation completed: nine offline checks passed in standard and isolated Python modes; the live script help and syntax checked successfully; bundled label coverage, unchanged response handling, and confidence-preserving live summaries are verified without network calls. The shared client matches the source branch byte-for-byte. The source notebook's already executed live findings are documented; no new live calls were required for this packaging update.
