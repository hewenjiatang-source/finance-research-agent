"""
Environment variable configuration loader

A unified wrapper around loading .env / .env.local, used by all modules.
Design principles:
  1. Secrets (API keys, base URLs) are read only from .env and never hard-coded in the source.
  2. Constructor arguments are only overrides of .env, convenient for unit tests and special cases.
  3. Idempotent loading: repeated calls do not re-read the files.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv


__all__ = ["ensure_env_loaded", "get_env", "get_env_int", "get_env_float", "get_env_bool"]


_ENV_LOADED = False


def ensure_env_loaded() -> None:
    """Make sure the .env files are loaded (idempotent).

    Load order (later loads take priority):
      1. .env (project-level defaults)
      2. .env.local (user-local customization, ignored by .gitignore)
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    # 1. load the project-level .env
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)

    # 2. load the user-level .env.local (higher priority)
    local_env = os.path.join(os.getcwd(), ".env.local")
    if os.path.exists(local_env):
        load_dotenv(dotenv_path=local_env, override=True)

    _ENV_LOADED = True


def get_env(key: str, default: str | None = None) -> str | None:
    """Read an environment variable; empty strings become None.

    The first call automatically triggers ensure_env_loaded().
    """
    ensure_env_loaded()
    val = os.getenv(key, default)
    if val == "" or val is None:
        return None
    return val


def get_env_int(key: str, default: int) -> int:
    """Read an environment variable and convert it to int."""
    val = get_env(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"The value '{val}' of environment variable {key} cannot be converted to an integer")


def get_env_float(key: str, default: float) -> float:
    """Read an environment variable and convert it to float."""
    val = get_env(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        raise ValueError(f"The value '{val}' of environment variable {key} cannot be converted to a float")


def get_env_bool(key: str, default: bool = False) -> bool:
    """Read an environment variable and convert it to bool.

    The following values count as True: true, True, 1, yes, YES
    The following values count as False: false, False, 0, no, NO
    """
    val = get_env(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")
