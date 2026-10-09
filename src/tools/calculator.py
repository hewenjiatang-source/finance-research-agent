"""
Calculator tool (CalculatorTool)

Rationale:
  code_sandbox can run arbitrary Python calculations, but:
  1. It carries security risk (needs AST checks)
  2. It starts slowly (even simple addition goes through the full sandbox flow)
  3. It is costly for the LLM to call (a full piece of Python code must be written)

  CalculatorTool offers lightweight, safe, fast deterministic computation:
  - Evaluates math expressions directly (no code-execution risk)
  - Supports unit conversion, percentage calculation, statistical functions
  - Low call cost: the Agent only passes an expression string

Relationship with code_sandbox:
  calculator:   simple arithmetic (2+2, 15% of 300, average([1,2,3]))
  code_sandbox: complex logic (data analysis, simulation, algorithm implementation)
"""
from __future__ import annotations

import ast
import math
import operator
import re
import statistics
from typing import Any


__all__ = ["CalculatorTool"]

# Allowed safe operators and functions
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCS = {
    "abs": abs,
    "round": round,
    "max": max,
    "min": min,
    "sum": sum,
    "len": len,
    # Math functions
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    # Statistical functions
    "mean": statistics.mean,
    "median": statistics.median,
    "stdev": statistics.stdev,
    "variance": statistics.variance,
    # Constants
    "pi": math.pi,
    "e": math.e,
}


class CalculatorTool:
    """Lightweight calculator: safely evaluates math expressions."""

    name: str = "calculator"
    description: str = (
        "Evaluate a mathematical expression safely. "
        "Use this for quick calculations instead of code_sandbox. "
        "Supports: +, -, *, /, **, %, abs, round, sqrt, sin, cos, log, mean, median, etc. "
        "Input: {'expression': str}. Output: result as string."
    )

    def get_openai_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate, e.g. '(150 + 230) * 0.15' or 'mean([12, 15, 18, 21])'",
                        },
                    },
                    "required": ["expression"],
                },
            },
        }

    async def execute(self, expression: str) -> str:
        """Evaluate a math expression.

        Args:
            expression: math expression string, e.g. "(150 + 230) * 0.15" or "mean([12, 15, 18, 21])".

        Returns:
            String representation of the result.
        """
        if not expression or not expression.strip():
            return "[Calculator Error] Empty expression"

        # Simulate async IO (the computation is CPU-bound, but keep the interface consistent)
        import asyncio
        await asyncio.sleep(0)

        try:
            # Preprocessing: convert full-width brackets, percent signs, etc. to the standard format
            expr = self._preprocess(expression)
            result = self._safe_eval(expr)
            return f"{result}"
        except ZeroDivisionError:
            return "[Calculator Error] Division by zero"
        except ValueError as e:
            return f"[Calculator Error] Invalid value: {e}"
        except Exception as e:
            return f"[Calculator Error] {type(e).__name__}: {e}"

    @staticmethod
    def _preprocess(expr: str) -> str:
        """Preprocess the expression: unify the format."""
        # Full-width brackets -> ASCII brackets (Chinese characters below are intentional)
        expr = expr.replace("（", "(").replace("）", ")")
        expr = expr.replace("【", "[").replace("】", "]")
        # Percent sign handling: 15% -> 15/100
        expr = re.sub(r"(\d+(?:\.\d+)?)%", r"(\1/100)", expr)
        # Strip thousands-separator commas (only commas between digits, e.g. 1,000 -> 1000; list commas are kept)
        expr = re.sub(r"(\d),(?=\d)", r"\1", expr)
        return expr.strip()

    def _safe_eval(self, expr: str) -> Any:
        """Safe eval: only math AST nodes are allowed."""
        tree = ast.parse(expr, mode="eval")
        return self._eval_node(tree.body)

    def _eval_node(self, node: ast.AST) -> Any:
        """Recursively evaluate AST nodes."""
        if isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        if isinstance(node, ast.Constant):  # Python >= 3.8
            return node.value
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _SAFE_OPS:
                raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
            return _SAFE_OPS[op_type](self._eval_node(node.left), self._eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _SAFE_OPS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            return _SAFE_OPS[op_type](self._eval_node(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are allowed")
            func_name = node.func.id
            if func_name not in _SAFE_FUNCS:
                raise ValueError(f"Unsupported function: {func_name}")
            args = [self._eval_node(arg) for arg in node.args]
            return _SAFE_FUNCS[func_name](*args)
        if isinstance(node, ast.Name):
            if node.id not in _SAFE_FUNCS:
                raise ValueError(f"Unsupported name: {node.id}")
            return _SAFE_FUNCS[node.id]
        if isinstance(node, ast.List):
            return [self._eval_node(elt) for elt in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elt) for elt in node.elts)
        if isinstance(node, ast.Subscript):
            value = self._eval_node(node.value)
            slice_val = self._eval_node(node.slice)
            return value[slice_val]
        if isinstance(node, ast.Index):  # Python < 3.9 compatibility
            return self._eval_node(node.value)

        raise ValueError(f"Unsupported AST node: {type(node).__name__}")
