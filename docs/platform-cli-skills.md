# Platform CLI skills

## Purpose

Release 0.3.0 adds three standalone skills to the shared plugin catalog:

| Skill | Use it for | Expected output |
| --- | --- | --- |
| `cf-cli` | Cloud Foundry target discovery, app and service diagnosis, manifest review, and deployment preparation | Scoped findings, local readiness checks, and an inspected command for the user to run when deployment is needed |
| `btp-cli` | BTP account, entitlement, service, access, environment, and Destination operations | Targeted readback or a reviewed, account-scoped change with its effects and verification |
| `joule-cli` | Joule Studio CLI installation and tenant checks, capability lint/compile, assistant inspection, and deployment preparation | Validated design-time artifacts, an inspected deployment command for the user, and a separate runtime verification plan |

The skills apply the useful operational scope from SAP's internal A2A Agent
Toolkit while using the current `@sap/joule-studio-cli` package and installed CLI help.
They avoid examples that expose secrets in arguments or output. They also keep
application and Joule deployments user-run, per this repository's conventions.

## Inputs and boundaries

- `cf-cli` needs the intended CF API, org, space, app or service, plus the
  application's manifest and local build context if deployment is involved.
- `btp-cli` needs the intended global account or subaccount ID, region, resource
  identity, and a CLI session authorized for that scope. It confirms the live
  action syntax before changing state.
- `joule-cli` needs a supported Node.js version, installed Joule Studio CLI,
  intended tenant, and an assistant project rooted at `da.sapdas.yaml` for
  linting and compilation.

All three skills distinguish local inspection from live verification. They do
not treat successful compilation, a CLI response, or a created Destination as
proof that an A2A agent works in Joule. The user submits `cf push`, `cf deploy`,
`joule deploy`, and `joule update` manually.

For an existing A2A agent being connected to Joule, apply
`sap-agent-joule-cf-bootstrap` alongside the relevant CLI skill. That skill owns
the A2A 0.3 contract, Cloud Foundry ingress, OAuth2 Destination mapping, and
Joule capability artifacts. If no agent exists, its workflow delegates agent
creation to `create-langgraph-react-agent`.

## Related files

- `plugins/sap-ai4u-skills/skills/cf-cli/SKILL.md`
- `plugins/sap-ai4u-skills/skills/btp-cli/SKILL.md`
- `plugins/sap-ai4u-skills/skills/joule-cli/SKILL.md`
- `plugins/sap-ai4u-skills/skills/sap-agent-joule-cf-bootstrap/SKILL.md`
- `plugins/sap-ai4u-skills/skills/sap-btp-ai/SKILL.md` for AI-task routing
- The three manifests under `plugins/sap-ai4u-skills/` and its `CHANGELOG.md`

## Validation

Run the Skill Creator validator for each new entry and then validate the
complete package:

```bash
python3 "<path-to-skill-creator>/scripts/quick_validate.py" plugins/sap-ai4u-skills/skills/cf-cli
python3 "<path-to-skill-creator>/scripts/quick_validate.py" plugins/sap-ai4u-skills/skills/btp-cli
python3 "<path-to-skill-creator>/scripts/quick_validate.py" plugins/sap-ai4u-skills/skills/joule-cli
python3 scripts/validate_package.py
python3 -m unittest discover -s plugins/sap-ai4u-skills/skills/sap-agent-joule-cf-bootstrap/tests
```

The first three checks validate skill metadata and structure. The package
validator checks manifests and all top-level skills. The existing Joule tests
check the integration boundary and package compatibility. Live BTP and Joule
operations need an authorized target tenant and are intentionally outside this
repository validation.
