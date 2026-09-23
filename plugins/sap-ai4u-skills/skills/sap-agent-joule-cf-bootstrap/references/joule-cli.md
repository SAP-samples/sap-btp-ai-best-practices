# Joule Studio CLI Lifecycle

Use this reference to validate design-time artifacts and guide the user through assistant deployment and management.

## Install and Authenticate

Use a supported Node.js version in the range 20.12 through 24 and the current package:

```bash
node --version
npm install -g @sap/joule-studio-cli
joule --version
joule login
joule status
```

Do not substitute the older `@sap/joule-cli` package. Run `joule <command> --help` on the installed version before relying on flags because CLI behavior can move independently of this skill.

Use the user's tenant URL and approved interactive authentication flow. Do not store Joule credentials in project files.

Required tenant role collections can vary with landscape and product rollout. Verify them in the current SAP Help for the tenant instead of hard-coding a possibly stale list. Missing authorization is a tenant prerequisite, not a YAML compile error.

## Descriptor and Capability Constraints

- Run commands from the directory containing `da.sapdas.yaml`.
- Use descriptor schema `1.4.0` and capability schema `3.28.0` for this flow.
- Put custom capabilities under namespace `joule.ext`.
- Keep function and scenario names at or below 30 characters.
- Keep assistant names between 3 and 50 letters, digits, or underscores, without consecutive underscores.
- Keep the compiled package below the tenant's current size limit; 50 MB is the documented baseline.
- Keep scenario `response_context` to one section, no more than 10 root-level fields, and no more than 10,000 bytes.

## Local Validation Is Not Deployment

Run:

```bash
joule lint
joule compile
```

`joule lint` finds design-time consistency problems. `joule compile` builds the assistant package. Neither command verifies the deployed CF route, OAuth client, BTP Destination, tenant roles, or runtime behavior.

If available in the installed version, use `joule link` for its documented local authoring flow. Do not describe a command as a test command unless `joule --help` actually exposes it.

## Manual Deployment

Only the user executes state-changing lifecycle commands. Inspect their help first:

```bash
joule deploy --help
joule update --help
joule launch --help
joule get --help
joule list --help
joule delete --help
joule remove --help
```

Typical first deployment starts with the compiled project from the directory containing `da.sapdas.yaml`:

```bash
joule deploy -c -n "<assistant-name>"
```

Use the flags shown by the installed CLI if they differ. Deployment can take substantially longer than compilation; the documented baseline timeout is up to 60 minutes.

Use:

- `joule list` to inventory assistants before selecting a target;
- `joule get` to inspect one assistant;
- `joule launch` to open the deployed assistant for end-to-end validation;
- `joule update` for a deliberate update after diff and compile review;
- `joule delete` or `joule remove` only after resolving exact semantics and target identifiers from `--help`.

Deletion and removal may affect users and may not be recoverable without a redeploy. Never infer authorization for them from a request to validate or troubleshoot.

## Troubleshooting Order

Keep failures in their own layer:

1. `joule lint`: schema, naming, references, and static artifact issues.
2. `joule compile`: packaging or design-time compilation issues.
3. `joule deploy`/`update`: tenant authentication, authorization, package limits, and deployment state.
4. launch-time failure: Destination alias, Destination auth, remote route, A2A response, or runtime timeout.

For launch-time errors, validate the CF route and BTP Destination independently before changing the Joule YAML.

## Sources

- [SAP Help: Code-based agents](https://help.sap.com/docs/joule/service-guide/code-based-agents)
