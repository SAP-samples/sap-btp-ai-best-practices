"""Offline self-check for the A2A connector MCP server (standard library only).

Examples:
    python3 plugins/ai4u-a2a-connector/server/test_connector.py

What it does:
    1. Unit-checks URL handling, auth building, and A2A result flattening.
    2. Starts a local fake A2A agent (HTTP, API-key protected) and a fake
       `btp` script that lists it as a destination.
    3. Runs `mcp_server.py` as a subprocess and drives it over stdio:
       initialize, tools/list, list_agents, send_message (with context
       continuity and an input-required turn), and error paths.
    4. Asserts that no destination secret ever reaches the MCP output.

No network access, BTP session, or real agent is needed. Exits non-zero on
the first failed assertion.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
# The package validator rejects __pycache__ folders inside plugins.
sys.dont_write_bytecode = True
sys.path.insert(0, str(SERVER_DIR))

from a2a_client import card_candidates, flatten_result, same_host  # noqa: E402
from destinations import Destination, request_auth  # noqa: E402

API_KEY = "secret-key-123"


def check_units() -> None:
    """Check pure helpers: card paths, host guard, auth, and result flattening."""

    assert card_candidates("https://h.example/") == [
        "https://h.example/.well-known/agent-card.json",
        "https://h.example/.well-known/agent.json",
        "https://h.example/a2a/.well-known/agent-card.json",
        "https://h.example/a2a/.well-known/agent.json",
    ]
    assert card_candidates("https://h.example/a2a") == [
        "https://h.example/a2a/.well-known/agent-card.json",
        "https://h.example/a2a/.well-known/agent.json",
        "https://h.example/.well-known/agent-card.json",
        "https://h.example/.well-known/agent.json",
    ]
    assert same_host("https://h.example/a2a/", "https://h.example")
    assert not same_host("https://evil.example/a2a/", "https://h.example")
    assert not same_host("http://h.example/", "https://h.example")

    basic = Destination(
        "B", "https://h.example", "BasicAuthentication", "",
        {"User": "u", "Password": "p", "URL.headers.X-API-Key": "k", "URL.queries.sap-client": "100"},
    )
    headers, query = request_auth(basic)
    assert headers == {"X-API-Key": "k", "Authorization": "Basic dTpw"}
    assert query == {"sap-client": "100"}
    assert "Password" not in repr(basic) and "X-API-Key" not in repr(basic), "config leaked into repr"

    completed = {"result": {"id": "t1", "contextId": "c1", "status": {"state": "completed"},
                            "artifacts": [{"parts": [{"kind": "text", "text": "done"}]}]}}
    assert flatten_result(completed) == {"state": "completed", "text": "done", "context_id": "c1", "task_id": None}
    waiting = {"result": {"id": "t2", "contextId": "c1", "status": {
        "state": "input-required", "message": {"parts": [{"kind": "text", "text": "Which one?"}]}}}}
    assert flatten_result(waiting) == {"state": "input-required", "text": "Which one?", "context_id": "c1", "task_id": "t2"}
    failed = {"result": {"id": "t3", "status": {"state": "failed",
                                                "message": {"parts": [{"kind": "text", "text": "Agent request failed"}]}}}}
    assert flatten_result(failed)["text"] == "Agent request failed"
    message = {"result": {"kind": "message", "contextId": "c9", "parts": [{"kind": "text", "text": "hi"}]}}
    assert flatten_result(message) == {"state": "completed", "text": "hi", "context_id": "c9", "task_id": None}
    error = {"error": {"code": -32602, "message": "bad"}}
    assert flatten_result(error)["state"] == "error"


class FakeAgent(BaseHTTPRequestHandler):
    """Minimal A2A 0.3 agent: card under /a2a (API key required) and an open card under /open."""

    def log_message(self, *args: object) -> None:
        """Silence the default per-request logging."""

    def _reply(self, status: int, body: dict | None = None) -> None:
        """Write a JSON (or empty) response."""

        data = json.dumps(body).encode() if body is not None else b""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _card(self, prefix: str) -> dict:
        """Build an agent card whose JSON-RPC URL sits under `prefix`."""

        host = f"http://127.0.0.1:{self.server.server_address[1]}"
        return {"name": "fake-agent", "description": "Test agent", "version": "1.0.0",
                "protocolVersion": "0.3", "url": f"{host}{prefix}/a2a/",
                "skills": [{"id": "s", "name": "Echo", "description": "Echoes", "examples": ["hello"]}]}

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        """Serve agent cards; everything else is 404."""

        if self.path == "/a2a/.well-known/agent-card.json":
            if self.headers.get("X-API-Key") != API_KEY:
                return self._reply(401, {"detail": "missing key"})
            return self._reply(200, self._card(""))
        if self.path == "/open/a2a/.well-known/agent-card.json":
            return self._reply(200, self._card("/open"))
        return self._reply(404, {"detail": "Not Found"})

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        """Answer `message/send`: echo, or ask a question when told 'ask me'."""

        if self.headers.get("X-API-Key") != API_KEY:
            return self._reply(401, {"detail": "missing key"})
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        message = request["params"]["message"]
        assert request["method"] == "message/send" and message["parts"][0]["kind"] == "text"
        text = message["parts"][0]["text"]
        context = message.get("contextId") or "ctx-new"
        if text == "ask me":
            result = {"kind": "task", "id": "task-q", "contextId": context, "status": {
                "state": "input-required", "message": {"parts": [{"kind": "text", "text": "Which project?"}]}}}
        else:
            answer = f"echo: {text} (task {message.get('taskId')})"
            result = {"kind": "task", "id": "task-a", "contextId": context, "status": {"state": "completed"},
                      "artifacts": [{"name": "agent_result", "parts": [{"kind": "text", "text": answer}]}]}
        self._reply(200, {"jsonrpc": "2.0", "id": request["id"], "result": result})


def write_fake_btp(folder: Path, port: int) -> Path:
    """Write a fake `btp` executable that prints a destination list and return its path."""

    base = f"http://127.0.0.1:{port}"
    destinations = [
        {"destinationConfiguration": {"Name": "FAKE_AGENT", "Type": "HTTP", "ProxyType": "Internet",
                                      "URL": base, "Authentication": "NoAuthentication",
                                      "URL.headers.X-API-Key": API_KEY}},
        {"destinationConfiguration": {"Name": "USER_EXCHANGE_AGENT", "Type": "HTTP", "ProxyType": "Internet",
                                      "URL": f"{base}/open", "Authentication": "OAuth2UserTokenExchange"}},
        {"destinationConfiguration": {"Name": "NOT_AN_AGENT", "Type": "HTTP", "ProxyType": "Internet",
                                      "URL": f"{base}/nothing", "Authentication": "NoAuthentication"}},
        {"destinationConfiguration": {"Name": "DEAD_HOST", "Type": "HTTP", "ProxyType": "Internet",
                                      "URL": "http://127.0.0.1:9", "Authentication": "NoAuthentication"}},
        {"destinationConfiguration": {"Name": "ON_PREMISE", "Type": "HTTP", "ProxyType": "OnPremise",
                                      "URL": base, "Authentication": "NoAuthentication"}},
    ]
    listing = folder / "destinations.json"
    listing.write_text(json.dumps(destinations))
    script = folder / "btp"
    script.write_text(f"#!/bin/sh\ncat '{listing}'\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def check_server(btp_path: Path) -> None:
    """Drive mcp_server.py over stdio and assert on every tool path."""

    env = {**os.environ, "BTP_CLI_PATH": str(btp_path)}
    process = subprocess.Popen(
        [sys.executable, "-B", str(SERVER_DIR / "mcp_server.py")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env,
    )
    transcript: list[str] = []
    counter = iter(range(1, 1000))

    def rpc(method: str, params: dict | None = None) -> dict:
        """Send one request line and read one response line."""

        process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": next(counter), "method": method,
                                        "params": params or {}}) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        transcript.append(line)
        return json.loads(line)

    def tool(name: str, **arguments: object) -> tuple[bool, object]:
        """Call a tool and return (is_error, parsed JSON or raw text)."""

        result = rpc("tools/call", {"name": name, "arguments": arguments})["result"]
        text = result["content"][0]["text"]
        return bool(result.get("isError")), (text if result.get("isError") else json.loads(text))

    try:
        init = rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                  "clientInfo": {"name": "test", "version": "0"}})
        assert init["result"]["protocolVersion"] == "2025-06-18"
        process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        assert [t["name"] for t in rpc("tools/list")["result"]["tools"]] == [
            "list_agents", "get_agent_card", "send_message"]
        assert rpc("unknown/method")["error"]["code"] == -32601

        is_error, listing = tool("list_agents")
        assert not is_error, listing
        names = {a["agent"]: a for a in listing["agents"]}
        assert set(names) == {"FAKE_AGENT", "USER_EXCHANGE_AGENT"}, names
        assert names["FAKE_AGENT"]["callable"] and not names["USER_EXCHANGE_AGENT"]["callable"]
        assert names["FAKE_AGENT"]["skills"][0]["examples"] == ["hello"]
        assert listing["destinations_scanned"] == 4  # on-premise destination excluded

        is_error, card = tool("get_agent_card", agent="fake_agent")
        assert not is_error and card["name"] == "fake-agent"

        is_error, first = tool("send_message", agent="FAKE_AGENT", message="hello")
        assert not is_error and first["state"] == "completed" and first["context_id"] == "ctx-new", first
        is_error, question = tool("send_message", agent="FAKE_AGENT", message="ask me",
                                  context_id=first["context_id"])
        assert question == {"state": "input-required", "text": "Which project?",
                            "context_id": "ctx-new", "task_id": "task-q"}, question
        is_error, answer = tool("send_message", agent="FAKE_AGENT", message="the first",
                                context_id=question["context_id"], task_id=question["task_id"])
        assert answer["text"] == "echo: the first (task task-q)" and answer["task_id"] is None, answer

        is_error, text = tool("send_message", agent="USER_EXCHANGE_AGENT", message="hi")
        assert is_error and "OAuth2UserTokenExchange" in text, text
        is_error, text = tool("send_message", agent="MISSING", message="hi")
        assert is_error and "Known agents: FAKE_AGENT, USER_EXCHANGE_AGENT" in text, text
        is_error, text = tool("send_message", agent="FAKE_AGENT", message="")
        assert is_error and "message" in text, text
    finally:
        process.stdin.close()
        process.wait(timeout=10)
    assert API_KEY not in "".join(transcript), "destination secret leaked into MCP output"


def main() -> None:
    """Run the unit checks, then the end-to-end stdio check against local fakes."""

    check_units()
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeAgent)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory() as folder:
            check_server(write_fake_btp(Path(folder), server.server_address[1]))
    finally:
        server.shutdown()
    print("All connector checks passed.")


if __name__ == "__main__":
    main()
