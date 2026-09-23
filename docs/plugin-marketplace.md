# SAP AI4U plugin marketplace

## Purpose

This repository distributes one `sap-ai4u-skills` plugin containing 33 reusable
SAP BTP AI skills. Codex and Claude Code use separate marketplace catalogs at
the repository root; both catalogs point to the same plugin package under
`plugins/sap-ai4u-skills/`. No SAP BTP account is needed to install the skills.
Individual workflows may require SAP services and credentials when used.

The package was copied from the SAP AI4U plugin source at commit
`27e12f44708a42b261765f64eb7552997679b6b2`, version `0.3.0`.

## Inputs and outputs

| Input | Output |
| --- | --- |
| Git access to this public repository and a supported Codex or Claude Code client | The `sap-ai4u-skills@sap-ai4u` plugin installed with 33 skills |
| A marketplace refresh after a new package version | Updated plugin files in the client's plugin cache |

## Files

| Path | Role |
| --- | --- |
| `.agents/plugins/marketplace.json` | Codex marketplace catalog and install policy |
| `.claude-plugin/marketplace.json` | Claude Code marketplace catalog |
| `plugins/sap-ai4u-skills/plugin.json` | Portable plugin identity and version |
| `plugins/sap-ai4u-skills/.codex-plugin/plugin.json` | Codex compatibility metadata |
| `plugins/sap-ai4u-skills/.claude-plugin/plugin.json` | Claude Code compatibility metadata |
| `plugins/sap-ai4u-skills/skills/` | The skills and their supporting files |

The marketplace catalogs must remain at the Git repository root for ordinary
GitHub repository registration. Their relative source paths are resolved from
that root, so both point to `./plugins/sap-ai4u-skills`.

## Install

Codex:

```bash
codex plugin marketplace add SAP-samples/sap-btp-ai-best-practices \
  --sparse .agents/plugins --sparse plugins
codex plugin add sap-ai4u-skills@sap-ai4u
```

Claude Code:

```bash
claude plugin marketplace add SAP-samples/sap-btp-ai-best-practices \
  --sparse .claude-plugin plugins
claude plugin install sap-ai4u-skills@sap-ai4u
```

Start a new session after installing. The sparse options keep the checkout
focused on the root catalog and plugin package; they do not change where the
marketplace catalogs are discovered.

## Update and validate

After changing the plugin, update its version in all three plugin manifests and
record the change in `plugins/sap-ai4u-skills/CHANGELOG.md`. Keep both marketplace
source paths aligned with the package directory. Validate from this repository
root:

```bash
python3 -m json.tool .agents/plugins/marketplace.json >/dev/null
python3 -m json.tool .claude-plugin/marketplace.json >/dev/null
claude plugin validate .
```

To refresh an existing installation after a published change:

```bash
codex plugin marketplace upgrade sap-ai4u
codex plugin add sap-ai4u-skills@sap-ai4u
claude plugin marketplace update sap-ai4u
claude plugin update sap-ai4u-skills@sap-ai4u
```

OpenCode and Cline consume the package's `skills/` directory through their
skill loaders; they do not use either marketplace catalog.
