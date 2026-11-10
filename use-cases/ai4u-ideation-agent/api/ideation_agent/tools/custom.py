"""Custom LangGraph tools.

This module contains locally-defined tools that run within the agent process.
Add new custom tools here by defining functions decorated with @tool.

Example:
    @tool("my_tool")
    def my_tool(param: str) -> Dict[str, Any]:
        '''Tool description for the LLM.'''
        return {"result": do_something(param)}
"""
from __future__ import annotations

import ast
from typing import Any, Dict

from langchain_core.tools import tool

try:
    from ..skill_catalog import load_skill_bundle
except ImportError:  # pragma: no cover
    from skill_catalog import load_skill_bundle


def _safe_eval_arithmetic(expression: str) -> float:
    """Safely evaluate an arithmetic expression.

    Parses the expression into an AST and evaluates only allowed operations.
    This prevents code injection while allowing basic math.

    Args:
        expression: A string containing an arithmetic expression.

    Returns:
        The numeric result of the expression.

    Raises:
        ValueError: If the expression contains disallowed components.
    """
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
    """Evaluate a basic arithmetic expression and return the numeric result.

    Supports: +, -, *, /, **, %, //
    Examples: "2 + 2", "(10 + 5) * 3", "2 ** 8"
    """
    expr = str(expression).strip()
    if not expr:
        raise ValueError("calculator: 'expression' is required")
    return {"expression": expr, "result": _safe_eval_arithmetic(expr)}


@tool("load_skill")
def load_skill_tool(skill_name: str) -> Dict[str, Any]:
    """Load full local instructions for a skill by exact skill ID.

    Returns the content of `SKILL.md` plus sibling `.md` and `.txt` files in
    the same skill folder. Matching is strict by folder name.
    """
    skill_id = str(skill_name).strip()
    if not skill_id:
        raise ValueError("load_skill: 'skill_name' is required")
    return load_skill_bundle(skill_id)


# List of all custom tools to be registered with the agent
CUSTOM_TOOLS = [calculator_tool, load_skill_tool]
