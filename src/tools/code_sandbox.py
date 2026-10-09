"""
Code sandbox tool

Simulates running Python code and returns stdout / stderr / the return value.
This is currently a safe simulated version: it only parses simple expressions or returns predefined results and does not truly execute arbitrary code.
In production it can be replaced with a Docker sandbox or a restricted subprocess.
"""
from __future__ import annotations

import ast
import asyncio
import io
import random
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any


__all__ = ["CodeSandboxTool"]


class CodeSandboxTool:
    """Code sandbox tool.

    Security policy:
      1. By default only pure expressions parsed by ast.parse are allowed (no function definitions, no imports)
      2. Dangerous builtins are filtered by a blacklist
      3. Timeout protection (implemented outside via asyncio.wait_for)
      4. builtins access is restricted during real execution
    """

    name: str = "code_sandbox"
    description: str = (
        "Execute Python code in a sandboxed environment. "
        "Input: {'code': str, 'timeout': int(optional, default=10)}. "
        "Output: {'stdout': str, 'stderr': str, 'return_value': Any, 'success': bool}."
    )

    # Blacklist of dangerous builtins
    _FORBIDDEN_NAMES = {
        "__import__", "open", "eval", "exec", "compile",
        "input", "raw_input", "reload", "exit", "quit",
        "os", "sys", "subprocess", "shutil", "socket",
    }

    def __init__(self, use_mock: bool = False) -> None:
        from ..utils.env_config import get_env_int

        self.use_mock = use_mock
        # Default timeout is read from .env for easy central adjustment
        self.default_timeout = get_env_int("CODE_SANDBOX_TIMEOUT", 10)

    def get_openai_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Execution timeout in seconds",
                            "default": 10,
                        },
                    },
                    "required": ["code"],
                },
            },
        }

    async def execute(self, code: str, timeout: int | None = None) -> dict[str, Any]:
        """Run Python code in the sandbox.

        Args:
            code: Python code string.
            timeout: timeout in seconds.

        Returns:
            Dict containing stdout, stderr, return_value, success.
        """
        actual_timeout = timeout if timeout is not None else self.default_timeout
        if self.use_mock:
            return await self._mock_execute(code)
        return await self._safe_execute(code, actual_timeout)

    async def _mock_execute(self, code: str) -> dict[str, Any]:
        """Mock mode: simulate common computation results."""
        await asyncio.sleep(random.randint(50, 200) / 1000.0)

        code_stripped = code.strip().lower()
        if "1+1" in code_stripped or "1 + 1" in code_stripped:
            return {
                "stdout": "2\n",
                "stderr": "",
                "return_value": 2,
                "success": True,
            }
        if "fibonacci" in code_stripped or "fib" in code_stripped:
            return {
                "stdout": "[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]\n",
                "stderr": "",
                "return_value": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34],
                "success": True,
            }
        return {
            "stdout": f"# Mock execution of:\n{code}\n",
            "stderr": "",
            "return_value": None,
            "success": True,
        }

    async def _safe_execute(self, code: str, timeout: int) -> dict[str, Any]:
        """Restricted execution mode."""
        # 1. Syntax check
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "stdout": "",
                "stderr": f"SyntaxError: {e}",
                "return_value": None,
                "success": False,
            }

        # 2. Static safety check: walk the AST looking for forbidden nodes
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                return {
                    "stdout": "",
                    "stderr": "SecurityError: import statements are not allowed",
                    "return_value": None,
                    "success": False,
                }
            if isinstance(node, ast.Call):
                # Check whether a blacklisted function is called
                if isinstance(node.func, ast.Name) and node.func.id in self._FORBIDDEN_NAMES:
                    return {
                        "stdout": "",
                        "stderr": f"SecurityError: '{node.func.id}' is forbidden",
                        "return_value": None,
                        "success": False,
                    }

        # 3. Execute in the restricted environment
        def _run() -> dict[str, Any]:
            safe_globals = {"__builtins__": {}}
            safe_locals: dict[str, Any] = {}
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            try:
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    result = eval(code, safe_globals, safe_locals)
                return {
                    "stdout": stdout_buf.getvalue(),
                    "stderr": stderr_buf.getvalue(),
                    "return_value": result,
                    "success": True,
                }
            except Exception:
                return {
                    "stdout": stdout_buf.getvalue(),
                    "stderr": traceback.format_exc(),
                    "return_value": None,
                    "success": False,
                }

        try:
            # Use asyncio's run_in_executor to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, _run),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return {
                "stdout": "",
                "stderr": f"TimeoutError: execution exceeded {timeout}s",
                "return_value": None,
                "success": False,
            }
