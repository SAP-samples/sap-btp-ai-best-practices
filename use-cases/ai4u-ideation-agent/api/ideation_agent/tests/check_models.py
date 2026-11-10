"""Quick check: which models are deployed and reachable in the AI Core proxy.

Run:
    cd api
    python -m ideation_agent.tests.check_models
"""
from __future__ import annotations

import asyncio
import os
import time

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
except Exception:
    pass

from langchain_core.messages import HumanMessage

from ideation_agent.common import make_llm

MODELS = [
    ("openai", "gpt-4.1"),
    ("openai", "gpt-4.1-nano"),
    ("openai", "gpt-4.1-mini"),
    ("openai", "gpt-5.4"),
    ("openai", "gpt-4o"),
    ("google", "gemini-2.5-flash"),
    ("google", "gemini-2.5-pro"),
    ("google", "gemini-2.0-flash"),
]


async def check_model(provider: str, model: str) -> dict:
    try:
        llm = make_llm(provider=provider, model_name=model, temperature=0.1, max_tokens=10)
        t0 = time.time()
        resp = await llm.ainvoke([HumanMessage(content="Say OK")])
        elapsed = time.time() - t0
        return {"provider": provider, "model": model, "status": "ok", "latency": round(elapsed, 2)}
    except Exception as e:
        return {"provider": provider, "model": model, "status": "fail", "error": str(e)[:80]}


async def main():
    print(f"Checking model deployments...\n")
    results = await asyncio.gather(*[check_model(p, m) for p, m in MODELS])

    print(f"  {'Provider':<10} {'Model':<22} {'Status':<6} {'Latency':<10} {'Error'}")
    print(f"  {'─' * 75}")
    for r in results:
        latency = f"{r['latency']}s" if r["status"] == "ok" else ""
        error = r.get("error", "") if r["status"] == "fail" else ""
        status = "✓" if r["status"] == "ok" else "✗"
        print(f"  {r['provider']:<10} {r['model']:<22} {status:<6} {latency:<10} {error}")

    available = [r for r in results if r["status"] == "ok"]
    print(f"\n  {len(available)}/{len(MODELS)} models available")


if __name__ == "__main__":
    asyncio.run(main())
