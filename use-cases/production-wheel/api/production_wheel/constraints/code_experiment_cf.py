"""Durable HANA broker and explicit CF-task controller for developer validation.

The controller receives a trusted CAPI callable; it never deploys applications.
The runner receives only an expiring per-job capability, never API/CF/HANA secrets.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import shlex
import time
import uuid
from urllib.parse import urlsplit

from .code_experiment import MAX_INPUT_BYTES, MAX_OUTPUT_BYTES, ExperimentError, _candidate_index, validate_result

KIND = "constraint_code_jobs"
TERMINAL = {"SUCCEEDED", "FAILED"}


def encode(value, limit=MAX_INPUT_BYTES):
    """Serialize finite JSON artifacts and enforce the transport byte budget."""
    data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(data) > limit:
        raise ExperimentError("artifact size limit exceeded")
    return data


def decode(data):
    """Decode JSON without allowing duplicate fields or ambiguous sparse coefficients."""
    def unique(pairs):
        """Build one JSON object while rejecting duplicate keys."""
        result = {}
        for key, value in pairs:
            if key in result:
                raise ExperimentError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique)


class ConstraintCodeBroker:
    """Store job metadata and bounded artifacts in the existing HANA repository."""

    def __init__(self, repository, clock=time.time):
        """Accept the application repository and an injectable clock for lifecycle tests."""
        self.repo, self.clock = repository, clock

    def create(self, code, payload, timeout_seconds=30):
        """Persist one developer-requested experiment; return its public job metadata."""
        if not isinstance(code, str) or not isinstance(payload, dict):
            raise ExperimentError("code and payload required")
        _candidate_index(payload.get("candidates"))
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 60:
            raise ExperimentError("timeout_seconds must be an integer between 1 and 60")
        packet = encode({"code": code, "payload": payload, "timeout_seconds": timeout_seconds})
        identifier = uuid.uuid4().hex
        job = {"id": identifier, "revision": 1, "status": "PENDING", "created_at": str(self.clock()),
               "timeout_seconds": timeout_seconds, "task_guid": None, "task_name": "constraint-" + identifier,
               "disposal_confirmed": False}
        with self.repo.transaction():
            self.repo.insert(KIND, identifier, job)
            self.repo.put_artifact(identifier, "request.json", packet)
        return job

    def claim(self, identifier, runner_guid):
        """Atomically journal launch ownership and mint one short-lived capability."""
        capability = secrets.token_urlsafe(32)
        with self.repo.transaction():
            job = self.repo.lock_entity(KIND, identifier)
            if job["status"] != "PENDING":
                raise ExperimentError("job already claimed")
            job = self.repo.cas(KIND, identifier, job["revision"], {
                "status": "DISPATCHING", "runner_guid": runner_guid,
                "capability_hash": hashlib.sha256(capability.encode()).hexdigest(),
                "deadline": self.clock() + 110, "input_consumed": False,
            })
        return job, capability

    def _authorized(self, identifier, capability):
        """Lock and verify one live capability; caller must hold a repository transaction."""
        job = self.repo.lock_entity(KIND, identifier)
        digest = hashlib.sha256(capability.encode()).hexdigest()
        if job["status"] not in {"DISPATCHING", "RUNNING"} or self.clock() >= job.get("deadline", 0) or not hmac.compare_digest(digest, job.get("capability_hash", "")):
            raise ExperimentError("expired or invalid job capability")
        return job

    def input(self, identifier, capability):
        """Deliver the bounded source/features once; atomically reject input replay."""
        with self.repo.transaction():
            job = self._authorized(identifier, capability)
            if job["input_consumed"]:
                raise ExperimentError("job input already consumed")
            self.repo.cas(KIND, identifier, job["revision"], {"input_consumed": True, "status": "RUNNING"})
            return decode(self.repo.artifact(identifier, "request.json"))

    def complete(self, identifier, capability, result):
        """Validate and persist one response, revoking the capability on success or error."""
        encoded = encode(result, MAX_OUTPUT_BYTES)
        with self.repo.transaction():
            job = self._authorized(identifier, capability)
            if not job["input_consumed"]:
                raise ExperimentError("input must be consumed before completion")
            if not isinstance(result, dict) or set(result) != {"status", "result"} or result["status"] not in {"success", "error"}:
                raise ExperimentError("invalid runner response")
            if result["status"] == "success":
                packet = decode(self.repo.artifact(identifier, "request.json"))
                validate_result(result["result"], packet["payload"]["candidates"])
            elif result["result"] is not None:
                raise ExperimentError("error responses must omit generated diagnostics")
            self.repo.put_artifact(identifier, "result.json", encoded)
            self.repo.cas(KIND, identifier, job["revision"], {
                "status": "COMPLETED" if result["status"] == "success" else "FAILED",
                "capability_hash": "",
            })

    def inspect(self, identifier):
        """Return safe lifecycle metadata and validated output only after task disposal."""
        job = self.repo.get(KIND, identifier)
        public = {key: value for key, value in job.items() if key != "capability_hash"}
        if job["status"] == "COMPLETED" and job["disposal_confirmed"]:
            public["result"] = decode(self.repo.artifact(identifier, "result.json"))["result"]
        return public


class ConstraintTaskController:
    """Launch/reconcile already staged CF tasks using trusted CAPI and durable ownership."""

    def __init__(self, broker, cf_request, runner_guid, api_guid, space_guid, broker_url):
        """Configure explicit app/space identities and HTTPS broker; do not discover targets."""
        parsed = urlsplit(broker_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ExperimentError("credential-free HTTPS broker URL required")
        for identifier in (runner_guid, api_guid, space_guid):
            uuid.UUID(identifier)
        if runner_guid == api_guid:
            raise ExperimentError("runner must be a separate application")
        self.broker, self.cf = broker, cf_request
        self.runner, self.api, self.space, self.url = runner_guid, api_guid, space_guid, broker_url.rstrip("/")

    def verify_runner(self):
        """Fail closed on bindings, app variables, routes, permanent instances or wrong space."""
        app = self.cf(f"/v3/apps/{self.runner}")
        if app["relationships"]["space"]["data"]["guid"] != self.space:
            raise ExperimentError("runner space mismatch")
        env = self.cf(f"/v3/apps/{self.runner}/env")
        if any(env.get(key) for key in ("environment_variables", "staging_env_json", "running_env_json")) or env.get("system_env_json", {}).get("VCAP_SERVICES"):
            raise ExperimentError("runner must have no application credentials or bindings")
        for endpoint in (f"/v3/service_credential_bindings?app_guids={self.runner}", f"/v3/apps/{self.runner}/routes"):
            collection = self.cf(endpoint)
            if collection.get("resources") or collection.get("pagination", {}).get("next"):
                raise ExperimentError("runner must be unbound and have no routes")
        processes = self.cf(f"/v3/apps/{self.runner}/processes")
        if processes.get("pagination", {}).get("next") or any(p["instances"] for p in processes["resources"]):
            raise ExperimentError("runner must have zero permanent instances")
        if not self.cf(f"/v3/apps/{self.runner}/droplets/current").get("guid"):
            raise ExperimentError("runner needs a staged droplet")

    def dispatch(self, identifier):
        """Journal before launch; an ambiguous launch retains ownership for reconciliation."""
        self.verify_runner()
        job, capability = self.broker.claim(identifier, self.runner)
        command = shlex.join(["python", "-m", "constraint_experiment", "--job-url",
                              self.url + "/" + identifier, "--capability=" + capability])
        task = self.cf(f"/v3/apps/{self.runner}/tasks", body={
            "name": job["task_name"], "command": command, "memory_in_mb": 256, "disk_in_mb": 256,
        })
        with self.broker.repo.transaction():
            current = self.broker.repo.lock_entity(KIND, identifier)
            self.broker.repo.cas(KIND, identifier, current["revision"], {"task_guid": task["guid"]})
        return self.broker.inspect(identifier)

    def reconcile(self, identifier):
        """Recover ambiguous launches, cancel deadlines and retain ownership until terminal."""
        repo = self.broker.repo
        job = repo.get(KIND, identifier)
        if job.get("runner_guid") != self.runner:
            raise ExperimentError("job does not belong to configured runner")
        guid = job.get("task_guid")
        if not guid:
            matches, endpoint = [], f"/v3/apps/{self.runner}/tasks?per_page=5000"
            for _ in range(20):
                listing = self.cf(endpoint)
                matches.extend(t for t in listing["resources"] if t["name"] == job["task_name"])
                link = listing.get("pagination", {}).get("next")
                if not link:
                    break
                parsed = urlsplit(link["href"])
                endpoint = parsed.path + ("?" + parsed.query if parsed.query else "")
                if parsed.path != f"/v3/apps/{self.runner}/tasks":
                    raise ExperimentError("unexpected task pagination target")
            else:
                raise ExperimentError("task recovery page limit exceeded")
            if len(matches) != 1:
                raise ExperimentError("launch unresolved; retain ownership and reconcile again")
            guid = matches[0]["guid"]
        task = self.cf(f"/v3/tasks/{guid}")
        if task["state"] not in TERMINAL and (self.broker.clock() >= job["deadline"] or job["status"] in {"COMPLETED", "FAILED"}):
            self.cf(f"/v3/tasks/{guid}/actions/cancel", body={})
        with repo.transaction():
            current = repo.lock_entity(KIND, identifier)
            changes = {"task_guid": guid, "task_state": task["state"], "disposal_confirmed": task["state"] in TERMINAL}
            if task["state"] in TERMINAL and current["status"] not in {"COMPLETED", "FAILED"}:
                changes.update(status="FAILED", capability_hash="")
            if self.broker.clock() >= job["deadline"] and current["status"] not in {"COMPLETED", "FAILED"}:
                changes.update(status="FAILED", capability_hash="")
            repo.cas(KIND, identifier, current["revision"], changes)
        return self.broker.inspect(identifier)

    def cancel(self, identifier):
        """Revoke job access and request disposal while retaining unresolved ownership."""
        with self.broker.repo.transaction():
            job = self.broker.repo.lock_entity(KIND, identifier)
            if job.get("runner_guid") != self.runner:
                raise ExperimentError("job does not belong to configured runner")
            self.broker.repo.cas(KIND, identifier, job["revision"], {
                "status": "FAILED", "deadline": self.broker.clock(), "capability_hash": "",
            })
        return self.reconcile(identifier)
