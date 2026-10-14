# BTP Destination Lifecycle

Use this reference to connect the deployed A2A base route to the system alias in the Joule capability.

## Required Mapping

The same value must appear in all three places:

```yaml
system_aliases:
  inventory-agent-a2a:
    destination: inventory-agent-a2a
```

and in the BTP subaccount as Destination name `inventory-agent-a2a`.

Destination properties for production:

| Property | Value |
| --- | --- |
| `Name` | Exact Joule system alias |
| `Type` | `HTTP` |
| `ProxyType` | `Internet` |
| `URL` | Deployed HTTPS API base route |
| `Authentication` | `OAuth2ClientCredentials` |
| `clientId` | OAuth client ID from the ingress identity service |
| `clientSecret` | OAuth client secret |
| `tokenServiceURL` | OAuth token endpoint |

The URL is the base route, for example `https://inventory-agent.example.com`. Do not append `/a2a`, `/`, `/.well-known/agent.json`, or a JSON-RPC method.

`NoAuthentication` is allowed only for an explicitly selected development-only destination and an intentionally non-production route.

## BTP CLI Preflight

The BTP CLI receives command definitions from its backend, so inspect the installed/current command contract:

```bash
btp --info
btp target
btp help create connectivity/destination
btp help update connectivity/destination
btp help get connectivity/destination
btp help delete connectivity/destination
```

Log in using the organization's approved flow, preferably SSO, and identify the exact subaccount ID. Passing `--subaccount` explicitly makes scripts and review output less ambiguous.

The native command shape is:

```bash
btp [OPTIONS] create connectivity/destination --configuration <JSON-or-file> --subaccount <ID>
btp [OPTIONS] update connectivity/destination --configuration <JSON-or-file> --subaccount <ID>
btp [OPTIONS] get connectivity/destination --name <NAME> --subaccount <ID>
btp [OPTIONS] delete connectivity/destination --name <NAME> --subaccount <ID>
```

Prefer `--format json` for readback and automation. Do not use `--verbose` when the configuration contains secrets because verbose CLI output can print file input.

## Safe Helper

The bundled helper is dry-run-first and uses OAuth2 by default:

```bash
export JOULE_AGENT_CLIENT_ID="<client-id>"
export JOULE_AGENT_CLIENT_SECRET="<client-secret>"
export JOULE_AGENT_TOKEN_URL="https://<authorization-server>/oauth/token"

python3 "<skill-path>/scripts/manage_destination.py" \
  --name "inventory-agent-a2a" \
  --url "https://inventory-agent.example.com" \
  --subaccount "<subaccount-id>"
```

The preview omits the three credential properties. After review, the user adds `--apply`. Apply performs:

1. native command capability check;
2. destination list/read to decide create or update;
3. mode-0600 temporary JSON creation;
4. create or update with no shell interpolation and no verbose mode;
5. temporary file removal;
6. destination readback.

Use this development-only variant only when explicitly requested:

```bash
python3 "<skill-path>/scripts/manage_destination.py" \
  --name "inventory-agent-a2a-dev" \
  --url "https://inventory-agent-dev.example.com" \
  --subaccount "<subaccount-id>" \
  --development-no-auth
```

## Manual or Cockpit Fallback

If `btp help create connectivity/destination` is unavailable, do not guess a legacy CLI syntax. Use the subaccount's Connectivity > Destinations cockpit UI or an approved Destination service API flow. Preserve the same security defaults, exact alias, base route, and credential handling.

Do not source an arbitrary project `.env` file, echo a service-key payload, print an access token, or put the client secret directly on a command line.

## Verification and Management

After create/update:

- read the Destination back at the exact subaccount scope;
- confirm Name, URL, Type, Proxy Type, and Authentication;
- confirm secrets exist without displaying their values;
- test the connection through the cockpit when available;
- validate an actual Joule invocation because readback alone does not test OAuth audience, token forwarding, or application middleware.

Before deletion, resolve the exact Destination and confirm it is not shared by another assistant. The deletion is recoverable only by recreating the configuration and credentials.

## Sources

- [SAP Help: BTP CLI command reference](https://help.sap.com/docs/btp/btp-cli-command-reference/btp-cli-command-reference)
- [SAP Help: update connectivity/destination](https://help.sap.com/docs/btp/btp-cli-command-reference/no-content-for-file-btp-update-connectivity-destination-md)
- [SAP Help: BTP CLI command syntax](https://help.sap.com/docs/btp/sap-business-technology-platform/command-syntax-of-btp-cli)
