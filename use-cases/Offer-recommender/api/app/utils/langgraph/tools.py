"""
LangChain-native tools for the API's LangGraph chatbot.

Implements one tool mirroring the tutorial examples:
- calculator: safely evaluates arithmetic expressions
"""

from __future__ import annotations

import ast
import time
import os
import httpx
from typing import Any, Dict, Optional, List

from langchain_core.tools import tool


def _safe_eval_arithmetic(expression: str) -> float:
    """Safely evaluate a basic arithmetic expression using Python's AST."""
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Load,
        ast.Mod,
        ast.FloorDiv,
    )

    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError("Disallowed expression component")

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant):
            return float(n.value)
        if isinstance(n, ast.Num):
            return float(n.n)
        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                return left / right
            if isinstance(n.op, ast.Pow):
                return left**right
            if isinstance(n.op, ast.Mod):
                return left % right
            if isinstance(n.op, ast.FloorDiv):
                return left // right
            raise ValueError("Unsupported operator")
        if isinstance(n, ast.UnaryOp):
            operand = _eval(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +operand
            if isinstance(n.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator")
        raise ValueError("Unsupported expression type")

    return _eval(tree)


@tool("calculator")
def calculator_tool(expression: str) -> Dict[str, Any]:
    """Evaluate an arithmetic `expression` like `2*(3+4)` and return the result."""
    expr = str(expression).strip()
    if not expr:
        raise ValueError("calculator: 'expression' is required")
    value = _safe_eval_arithmetic(expr)
    # Add a wait/delay
    time.sleep(1)
    return {"expression": expr, "result": value}


__all__ = [
    "calculator_tool",
]
