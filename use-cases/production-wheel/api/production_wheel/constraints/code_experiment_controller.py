"""Run/recover one explicitly submitted validation task through the current CF login.

Examples (local trusted controller only; never execute generated Python locally):
    PYTHONPATH=api python -m production_wheel.constraints.code_experiment_controller \
      --job-id ID --runner-guid UUID --api-guid UUID --space-guid UUID \
      --cf-api https://api.FOUNDATION --broker-url https://APP/api/constraint-code/jobs
    # Use the same command with --recover after interruption or an ambiguous launch.

HANA configuration comes from the existing application service. The CF login remains
local; only a one-job capability is included in the CF task command.
"""

import argparse
import json
import selectors
import subprocess
import time

from .code_experiment_cf import ConstraintCodeBroker, ConstraintTaskController


def capture(args):
    """Run trusted CF CLI calls with a five-second deadline and bounded diagnostic output."""
    process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    output = bytearray()
    deadline = time.monotonic() + 5
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map() or process.poll() is None:
                if time.monotonic() >= deadline:
                    raise RuntimeError("CF request deadline exceeded")
                for key, _ in selector.select(0.05):
                    chunk = key.fileobj.read1(8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    output.extend(chunk)
                    if len(output) > 2_000_000:
                        raise RuntimeError("CF response limit exceeded")
        if process.returncode:
            raise RuntimeError("CF request failed")
        return bytes(output)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdout.close()


def cf_request(endpoint, body=None):
    """Call CAPI through the current local login; never log commands or capabilities."""
    args = ["cf", "curl", endpoint]
    if body is not None:
        args += ["-X", "POST", "-d", json.dumps(body)]
    result = json.loads(capture(args))
    if result.get("errors"):
        raise RuntimeError("CAPI rejected request")
    return result


def main():
    """Dispatch/recover one durable job, show progress and confirm disposal before returning."""
    from app.workspace.dependencies import get_service

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("job-id", "runner-guid", "api-guid", "space-guid", "cf-api", "broker-url"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    import re
    actual = re.search(r"https?://[^\s]+", capture(["cf", "api"]).decode())
    if not actual or actual.group(0).rstrip("/") != args.cf_api.rstrip("/"):
        parser.exit(1, "CF foundation mismatch; target the approved foundation manually.\n")
    controller = ConstraintTaskController(ConstraintCodeBroker(get_service().repo), cf_request,
        args.runner_guid, args.api_guid, args.space_guid, args.broker_url)
    started = time.monotonic()
    try:
        if not args.recover:
            try:
                controller.dispatch(args.job_id)
            except Exception:
                print("Launch unresolved; reconciling durable ownership.", flush=True)
        while time.monotonic() - started < 130:
            state = controller.reconcile(args.job_id)
            elapsed = int(time.monotonic() - started)
            print(f"[{elapsed:3d}/130s] {state['status']} | CF {state.get('task_state', 'unresolved')}", flush=True)
            if state["disposal_confirmed"]:
                if state["status"] != "COMPLETED":
                    parser.exit(1, "Experiment failed; task disposal confirmed.\n")
                print(json.dumps(state["result"], allow_nan=False))
                return
            time.sleep(2)
        raise TimeoutError("controller deadline exceeded")
    finally:
        state = controller.broker.inspect(args.job_id)
        if not state["disposal_confirmed"] and state.get("runner_guid"):
            try:
                controller.cancel(args.job_id)
                for _ in range(4):
                    if controller.reconcile(args.job_id)["disposal_confirmed"]:
                        break
                    time.sleep(0.5)
            finally:
                if not controller.broker.inspect(args.job_id)["disposal_confirmed"]:
                    print("Disposal unconfirmed; durable ownership retained. Run --recover with the same target.", flush=True)


if __name__ == "__main__":
    main()
