"""Standalone PS4 pipe-drawing technical-feasibility workflow.

This package intentionally has no dependency on ``app.ecn.assistant`` or ECN
persistence.  It accepts a transient drawing upload, uses only its packaged
Markdown policy sources as LLM instructions, and returns an evidence-bearing
PS4 assessment.
"""
