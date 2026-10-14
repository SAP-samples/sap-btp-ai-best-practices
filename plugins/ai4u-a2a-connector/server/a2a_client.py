"""Discover A2A agents among BTP destinations and talk to them over A2A 0.3 JSON-RPC.

A destination counts as an agent when an A2A agent card can be fetched from
it. Agents are addressed by destination name (for example
`AI4U_PROJECT_LIST_AGENT`). Messages use the A2A 0.3 `message/send` method,
which is also the contract Joule uses, so every Joule-registered agent
supports it.

Environment variables:
    A2A_TIMEOUT_SECONDS  Timeout for one `message/send` call (default: 180).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from destinations import (
    SUPPORTED_AUTH,
    Destination,
    DestinationError,
    list_destinations,
    request_auth,
)

SEND_TIMEOUT = float(os.environ.get("A2A_TIMEOUT_SECONDS", "180"))
PROBE_TIMEOUT = 5.0
PROBE_WORKERS = 32
CARD_PATHS = ("/.well-known/agent-card.json", "/.well-known/agent.json")
# States where the agent waits for the caller; the task ID must be sent back.
PENDING_STATES = ("input-required", "auth-required")

# Discovery result for this process: lower-cased destination name -> Agent.
# ponytail: process-lifetime cache; `list_agents(refresh=true)` rebuilds it.
_agents: dict[str, "Agent"] | None = None
_scanned = 0


@dataclass(frozen=True)
class Agent:
    """A destination whose agent card was found.

    Attributes:
        destination: The BTP destination (holds the credentials).
        card: The agent card JSON as published by the agent.
        rpc_url: JSON-RPC endpoint for A2A 0.3 calls, on the destination host.
    """

    destination: Destination
    card: dict
    rpc_url: str


def http_json(
    url: str,
    headers: dict[str, str] | None = None,
    body: dict | None = None,
    timeout: float = PROBE_TIMEOUT,
) -> tuple[int, object]:
    """Send a GET (no body) or JSON POST (with body) and parse the JSON reply.

    Args:
        url: Absolute URL, including any query string.
        headers: Extra request headers.
        body: JSON payload; when given the request is a POST.
        timeout: Socket timeout in seconds.

    Returns:
        `(status, data)` where `data` is the parsed JSON or `None` for a
        non-JSON body. HTTP error statuses are returned, not raised.

    Raises:
        OSError: for connection-level failures (DNS, TLS, refused, timeout).
    """

    request_headers = {"Accept": "application/json", **(headers or {})}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=data, headers=request_headers, method="POST" if data else "GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, None


def with_query(url: str, query: dict[str, str]) -> str:
    """Return `url` with `query` parameters merged into its query string."""

    if not query:
        return url
    parts = urllib.parse.urlsplit(url)
    merged = urllib.parse.parse_qsl(parts.query) + list(query.items())
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(merged)))


def same_host(first: str, second: str) -> bool:
    """Return True when both URLs share scheme, host, and port.

    Destination credentials are only sent when this holds, so an agent card
    cannot redirect them to another server.
    """

    a, b = urllib.parse.urlsplit(first), urllib.parse.urlsplit(second)
    return (a.scheme, a.hostname, a.port) == (b.scheme, b.hostname, b.port)


def card_candidates(base_url: str) -> list[str]:
    """Return the agent-card URLs to try for a destination URL, in order.

    Destinations point either at the application root (card under `/a2a`)
    or directly at the A2A mount (`.../a2a`, card sometimes at the host
    root as the A2A spec suggests), so all three layouts are covered.
    """

    base = base_url.rstrip("/")
    parts = urllib.parse.urlsplit(base)
    origin = f"{parts.scheme}://{parts.netloc}"
    bases = [base] if base.endswith("/a2a") else [base, base + "/a2a"]
    bases += [origin] if origin not in bases else []
    return [prefix + path for prefix in bases for path in CARD_PATHS]


def is_card(data: object) -> bool:
    """Return True when `data` looks like an A2A agent card (0.3 or 1.0)."""

    return isinstance(data, dict) and "name" in data and (
        "url" in data or "supportedInterfaces" in data
    )


def rpc_url_from_card(card: dict, card_url: str) -> str:
    """Pick the JSON-RPC endpoint for A2A 0.3 calls from an agent card.

    Order: the 0.3 top-level `url`; else a `supportedInterfaces` entry with
    protocol version 0.3; else the first interface; else the directory that
    holds the `.well-known` card.
    """

    if isinstance(card.get("url"), str):
        return card["url"]
    interfaces = [i for i in card.get("supportedInterfaces") or [] if isinstance(i, dict)]
    for interface in interfaces:
        if str(interface.get("protocolVersion", "")).startswith("0.3") and interface.get("url"):
            return interface["url"]
    if interfaces and interfaces[0].get("url"):
        return interfaces[0]["url"]
    return card_url.split("/.well-known/")[0] + "/"


def probe_agent(destination: Destination) -> Agent | None:
    """Try to fetch an agent card for one destination.

    Cards are requested without credentials first; destination credentials
    are added only when the server answers 401/403. Connection failures end
    the probe for that destination immediately.

    Returns:
        The `Agent` when a card is found on the destination host, else `None`.
    """

    for card_url in card_candidates(destination.url):
        try:
            status, card = http_json(card_url)
            if status in (401, 403) and destination.auth in SUPPORTED_AUTH:
                headers, query = request_auth(destination)
                status, card = http_json(with_query(card_url, query), headers)
        except (OSError, DestinationError):
            return None
        if status == 200 and is_card(card):
            rpc_url = rpc_url_from_card(card, card_url)
            if not same_host(rpc_url, destination.url):
                print(
                    f"a2a-agents: skipping {destination.name}: card points to another host",
                    file=sys.stderr,
                )
                return None
            return Agent(destination, card, rpc_url)
    return None


def discover(refresh: bool = False) -> dict[str, Agent]:
    """Return the discovered agents, scanning the destinations when needed.

    Only `HTTP` destinations with `ProxyType: Internet` are probed; on-premise
    destinations are unreachable from a laptop.

    Args:
        refresh: Rebuild the cache even if a previous scan exists.
    """

    global _agents, _scanned
    if _agents is None or refresh:
        destinations = [
            d
            for d in list_destinations()
            if d.config.get("Type", "HTTP") == "HTTP"
            and d.config.get("ProxyType", "Internet") == "Internet"
        ]
        with ThreadPoolExecutor(PROBE_WORKERS) as pool:
            found = [agent for agent in pool.map(probe_agent, destinations) if agent]
        _agents = {agent.destination.name.lower(): agent for agent in found}
        _scanned = len(destinations)
    return _agents


def get_agent(name: str) -> Agent:
    """Return the agent for a destination name (case-insensitive).

    Rescans once when the name is unknown, so new destinations show up
    without restarting the server. Raises `DestinationError` listing the
    known agents when the name still does not match.
    """

    agent = discover().get(name.lower()) or discover(refresh=True).get(name.lower())
    if agent is None:
        known = ", ".join(sorted(a.destination.name for a in discover().values())) or "none"
        raise DestinationError(f"No A2A agent named '{name}'. Known agents: {known}.")
    return agent


def list_agents(refresh: bool = False) -> dict:
    """Summarize the discovered agents for the MCP client (no secrets, no URLs).

    Returns:
        `{"agents": [...], "destinations_scanned": int}` where each agent has
        its destination name, card name/description/version, authentication
        type, whether it is callable from here, and its advertised skills.
    """

    agents = discover(refresh)
    summaries = []
    for agent in sorted(agents.values(), key=lambda a: a.destination.name.lower()):
        card, destination = agent.card, agent.destination
        summaries.append(
            {
                "agent": destination.name,
                "name": card.get("name"),
                "description": card.get("description") or destination.description,
                "version": card.get("version"),
                "authentication": destination.auth,
                "callable": destination.auth in SUPPORTED_AUTH,
                "skills": [
                    {
                        "name": skill.get("name"),
                        "description": skill.get("description"),
                        "examples": skill.get("examples") or [],
                    }
                    for skill in card.get("skills") or []
                    if isinstance(skill, dict)
                ],
            }
        )
    return {"agents": summaries, "destinations_scanned": _scanned}


def get_agent_card(name: str) -> dict:
    """Return the public agent card of the named agent."""

    return get_agent(name).card


def _texts(parts: object) -> list[str]:
    """Return the text values of A2A parts, ignoring file and data parts."""

    if not isinstance(parts, list):
        return []
    return [p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)]


def flatten_result(response: dict) -> dict:
    """Reduce an A2A 0.3 JSON-RPC response to what a conversation needs.

    Args:
        response: The JSON-RPC response object (`result` or `error`).

    Returns:
        `{"state", "text", "context_id", "task_id"}`. `task_id` is only set
        while the agent waits for input (`input-required`/`auth-required`),
        mirroring the Joule contract: completed tasks are not resumed.
    """

    if "error" in response:
        error = response.get("error") or {}
        return {
            "state": "error",
            "text": f"JSON-RPC error {error.get('code')}: {error.get('message')}",
            "context_id": None,
            "task_id": None,
        }
    result = response.get("result") or {}
    if result.get("kind") == "message":
        return {
            "state": "completed",
            "text": "\n".join(_texts(result.get("parts"))),
            "context_id": result.get("contextId"),
            "task_id": None,
        }
    status = result.get("status") or {}
    state = status.get("state", "unknown")
    status_texts = _texts((status.get("message") or {}).get("parts"))
    artifact_texts = [
        text
        for artifact in result.get("artifacts") or []
        if isinstance(artifact, dict)
        for text in _texts(artifact.get("parts"))
    ]
    # A waiting agent asks its question in the status message; a finished one
    # returns its answer in artifacts. Fall back to the other when empty.
    if state in PENDING_STATES:
        texts = status_texts or artifact_texts
    else:
        texts = artifact_texts or status_texts
    return {
        "state": state,
        "text": "\n".join(texts),
        "context_id": result.get("contextId"),
        "task_id": result.get("id") if state in PENDING_STATES else None,
    }


def send_message(
    name: str, text: str, context_id: str | None = None, task_id: str | None = None
) -> dict:
    """Send one text message to an agent with A2A 0.3 `message/send`.

    Args:
        name: Destination name of the agent.
        text: The user message.
        context_id: Conversation ID returned by a previous call, to continue it.
        task_id: Task ID, only when the previous state was `input-required`.

    Returns:
        The flattened result from `flatten_result`.

    Raises:
        DestinationError: when the agent is unknown, not callable, unreachable,
            or does not answer with JSON-RPC.
    """

    agent = get_agent(name)
    headers, query = request_auth(agent.destination)
    message: dict = {
        "kind": "message",
        "messageId": uuid.uuid4().hex,
        "role": "user",
        "parts": [{"kind": "text", "text": text}],
    }
    if context_id:
        message["contextId"] = context_id
    if task_id:
        message["taskId"] = task_id
    payload = {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": "message/send",
        "params": {"message": message},
    }
    try:
        status, data = http_json(with_query(agent.rpc_url, query), headers, payload, SEND_TIMEOUT)
    except OSError as error:
        raise DestinationError(
            f"Request to agent '{agent.destination.name}' failed: {type(error).__name__}: {error}"
        ) from None
    if not isinstance(data, dict):
        raise DestinationError(
            f"Agent '{agent.destination.name}' answered HTTP {status} without a JSON-RPC body."
        )
    return flatten_result(data)
