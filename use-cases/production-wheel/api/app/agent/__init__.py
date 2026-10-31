"""Orchestrating ReAct agent (SAP Gen AI Hub) for the production-wheel optimizer.

The agent turns natural-language optimization intent into a validated, typed
SolveRequest, launches the solve as an async job, and explains the resulting
frontier. Import submodules directly (for example ``app.agent.runtime`` or
``app.agent.tools``); this package intentionally does not eagerly import the LLM
runtime, so the optimizer tools stay importable and testable without the full
model stack installed.
"""
