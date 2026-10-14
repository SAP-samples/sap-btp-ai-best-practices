# SAP AI4U Plugin Marketplace

This branch (`sap-ai4u-plugin`) of the SAP BTP AI best-practices repository distributes three plugins through the `sap-ai4u` marketplace:

| Plugin | Skills | Purpose |
| --- | --- | --- |
| `sap-ai4u-skills` | 33 | Curated skills for SAP BTP AI development |
| `ai4u-documentation-assistance` | 1 | Delivery documentation, starting with `technical-handover` |
| `ai4u-a2a-connector` | 1 + MCP server | Talk to A2A agents registered as BTP destinations |

Each plugin is installed independently. Codex and Claude Code use the repository marketplaces;
OpenCode and Cline consume the same Agent Skills through their native skill
loaders.

| Platform | Skills | This repository's marketplace manifest |
| --- | --- | --- |
| Codex | Supported | `.agents/plugins/marketplace.json` |
| Claude Code | Supported | `.claude-plugin/marketplace.json` |
| OpenCode | Supported | Not consumed; use `skills` or `skills.paths` |
| Cline | Supported | Not consumed; use `skills` |
| Agent Plugins 1.0 clients | Supported | Client-specific |

## Codex

Register the marketplace and then install the plugin:

```bash
codex plugin marketplace add \
  https://github.com/SAP-samples/sap-btp-ai-best-practices.git \
  --ref sap-ai4u-plugin
codex plugin add sap-ai4u-skills@sap-ai4u
codex plugin add ai4u-documentation-assistance@sap-ai4u
codex plugin add ai4u-a2a-connector@sap-ai4u
```

Refresh an existing installation:

```bash
codex plugin marketplace upgrade sap-ai4u
codex plugin add sap-ai4u-skills@sap-ai4u
codex plugin add ai4u-documentation-assistance@sap-ai4u
codex plugin add ai4u-a2a-connector@sap-ai4u
```

Start a new Codex task after installation or update so the plugin is loaded.

## Claude Code

Register the marketplace at user scope and install the plugin:

```bash
claude plugin marketplace add \
  'https://github.com/SAP-samples/sap-btp-ai-best-practices.git#sap-ai4u-plugin'
claude plugin install sap-ai4u-skills@sap-ai4u
claude plugin install ai4u-documentation-assistance@sap-ai4u
claude plugin install ai4u-a2a-connector@sap-ai4u
```

Refresh an existing installation:

```bash
claude plugin marketplace update sap-ai4u
claude plugin update sap-ai4u-skills@sap-ai4u
claude plugin update ai4u-documentation-assistance@sap-ai4u
claude plugin update ai4u-a2a-connector@sap-ai4u
```

Restart Claude Code after updating the plugin.

`ai4u-a2a-connector` also needs `python3` and a logged-in SAP BTP CLI
(`btp login --url https://cli.btp.cloud.sap --subdomain ai4u --sso`). For the
Claude Desktop app, upload it as a `.plugin` archive built as described in
[`docs/a2a-connector-plugin.md`](docs/a2a-connector-plugin.md).

## OpenCode and Cline

OpenCode and Cline can use the Agent Skills in this repository, but they do not
interpret the Codex or Claude marketplace manifests. Install all 35 skills
from the three plugins for both platforms at project scope with the open `skills` CLI:

```bash
npx -y skills add \
  https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/sap-ai4u-plugin \
  --skill '*' \
  --agent opencode cline \
  --copy
```

The `/tree/sap-ai4u-plugin` URL targets the plugin branch — the skills are **not** on
the repository's default branch. If your `skills` CLI version can't resolve a branch
from that URL, use the [direct checkout](#opencode-direct-checkout) below instead.

The `skills` CLI classifies OpenCode and Cline as **universal agents**. Universal
agents share one canonical directory, so selecting either agent or both does
not duplicate the skills:

- Project scope: `.agents/skills/`
- Global scope: `~/.agents/skills/`

The command above installs at project scope and creates `.agents/skills/` plus
`skills-lock.json`. Choose global scope in the installer prompt, or add
`--global`, to use `~/.agents/skills/` across projects. The explicit
`--agent opencode cline` records the intended consumers and avoids automatic
agent selection, even though both currently resolve to the same physical
directory.

Install only one skill by replacing `'*'` with its name, for example
`--skill sap-btp-ai`. Refresh managed copies with:

```bash
npx -y skills update -y
```

### OpenCode direct checkout

OpenCode can also read the canonical skill tree directly from a local clone. Clone the
plugin branch (`--single-branch` keeps it small), then add this to `opencode.json`,
replacing the placeholder with the real absolute path:

```bash
git clone --branch sap-ai4u-plugin --single-branch \
  https://github.com/SAP-samples/sap-btp-ai-best-practices.git
```

```json
{
  "$schema": "https://opencode.ai/config.json",
  "skills": {
    "paths": [
      "/absolute/path/to/sap-btp-ai-best-practices/plugins/sap-ai4u-skills/skills",
      "/absolute/path/to/sap-btp-ai-best-practices/plugins/ai4u-documentation-assistance/skills",
      "/absolute/path/to/sap-btp-ai-best-practices/plugins/ai4u-a2a-connector/skills"
    ]
  }
}
```

Restart OpenCode after changing its configuration. Subsequent `git pull`
operations update the skills without another installation step.

OpenCode's native plugins are JavaScript/TypeScript modules or npm packages;
it does not currently provide a Codex/Claude-style addable marketplace for
this package. No native OpenCode plugin is needed for a skill-only catalog.


## Other compatible clients

Clients implementing Agent Plugins 1.0 can consume the directories
`plugins/sap-ai4u-skills`, `plugins/ai4u-documentation-assistance`, and
`plugins/ai4u-a2a-connector`. The
portable standard defines the plugin package, not a universal marketplace or
update protocol.

## Maintenance

The Joule and Cloud Foundry lifecycle redesign is documented in
[`docs/joule-cf-lifecycle-skill.md`](docs/joule-cf-lifecycle-skill.md). The new
Cloud Foundry, BTP, and Joule CLI skills and their validation are documented in
[`docs/platform-cli-skills.md`](docs/platform-cli-skills.md). The
documentation plugin and the multi-plugin validation are documented in
[`docs/documentation-assistance-plugin.md`](docs/documentation-assistance-plugin.md).
The A2A connector plugin (MCP server and skill) is documented in
[`docs/a2a-connector-plugin.md`](docs/a2a-connector-plugin.md).

Run the repository validation before every release:

```bash
python3 -m pip install -r requirements-dev.txt
python3 scripts/validate_package.py
```
