"""
Multi-backend LLM router (Model Router)

Supports configuring several LLM backends via environment variables (.env) and switching at runtime:
  - Claude (Anthropic Messages API, the default backend; name = "claude" or "anthropic")
  - DeepSeek API / local vLLM / OpenAI and any OpenAI-compatible API (kept for comparison experiments)

Key points:
  1. Switch backends with zero source changes: all secrets (API keys / URLs) live in the .env file
  2. Hot switching at runtime: different modules can use different backends (e.g. a cheap model for the Red Agent, a strong one for the Solver)
  3. Backward compatible: keeps all VLLMPolicy interfaces and behavior, only extends initialization

Usage examples:
  >>> from src.models.model_router import ModelRouter
  >>> # use the default backend (set by DEFAULT_LLM_BACKEND in .env)
  >>> policy = ModelRouter.create_backend()
  >>> # specify a backend explicitly
  >>> policy = ModelRouter.create_backend("deepseek")
  >>> # get all available backends and assign them by scenario
  >>> backends = ModelRouter.get_all_backends()
  >>> solver_policy = backends["deepseek"]
  >>> red_policy = backends["vllm"]  # local model, low attack cost
"""
from __future__ import annotations

import json
import os
from typing import Union

from ..utils.env_config import ensure_env_loaded, get_env
from .claude_policy import ClaudePolicy, DEFAULT_CLAUDE_MODEL
from .vllm_policy import VLLMPolicy


__all__ = ["ModelRouter"]

# aliases of the Claude backend (all point to the Anthropic Messages API)
_CLAUDE_NAMES = ("claude", "anthropic")

# global cache, to avoid re-reading .env and re-creating clients
_BACKEND_CACHE: dict[str, Union[VLLMPolicy, ClaudePolicy]] = {}


class ModelRouter:
    """Multi-backend LLM router.

    All methods are class / static methods; no instantiation needed.
    """

    @staticmethod
    def create_backend(
        backend_name: str | None = None,
        **override_kwargs,
    ) -> Union[VLLMPolicy, ClaudePolicy]:
        """Create the named LLM backend (ClaudePolicy or VLLMPolicy, which share an interface).

        Args:
            backend_name: backend name, matching the prefix in .env.
                          When None, DEFAULT_LLM_BACKEND is used.
            **override_kwargs: override any parameter from .env (e.g. temperature, max_tokens).

        Returns:
            VLLMPolicy: a configured policy instance with exactly the same interface as project one.

        Raises:
            ValueError: when no configuration for the backend is found.
        """
        ensure_env_loaded()

        name = (backend_name or get_env("DEFAULT_LLM_BACKEND", "claude")).lower().strip()

        # check the cache (kwargs may contain dicts, which cannot be hashed directly)
        cache_key = f"{name}:{json.dumps(override_kwargs, sort_keys=True, default=str)}"
        if cache_key in _BACKEND_CACHE:
            return _BACKEND_CACHE[cache_key]

        # read the configuration from .env by name
        config = ModelRouter._load_backend_config(name)
        config.update(override_kwargs)

        # create the Policy instance: Claude uses the Anthropic SDK, the others the OpenAI-compatible client
        policy = ClaudePolicy(**config) if name in _CLAUDE_NAMES else VLLMPolicy(**config)
        _BACKEND_CACHE[cache_key] = policy
        return policy

    @staticmethod
    def get_all_backends(backend_names: list[str] | None = None) -> dict[str, Union[VLLMPolicy, ClaudePolicy]]:
        """Preload and return all configured backends.

        Args:
            backend_names: list of backend names to scan. When None, scan all known backends
                          (claude, deepseek, vllm, openai, mimo and any custom prefix).

        Typical for scenarios like "main model on DeepSeek, Red Agent on MiMo".
        """
        ensure_env_loaded()
        backends: dict[str, Union[VLLMPolicy, ClaudePolicy]] = {}

        # by default scan all known built-in backends + custom backends discovered in the environment
        if backend_names is None:
            backend_names = ["claude", "deepseek", "vllm", "openai", "mimo"]
            # auto-discover other custom backends in .env ending in _API_KEY (anthropic is an alias of claude, skip it)
            for key in os.environ:
                if key.endswith("_API_KEY"):
                    prefix = key[:-len("_API_KEY")].lower()
                    if prefix not in backend_names and prefix not in _CLAUDE_NAMES:
                        backend_names.append(prefix)

        for name in backend_names:
            if ModelRouter._is_backend_configured(name):
                try:
                    backends[name] = ModelRouter.create_backend(name)
                except ValueError:
                    pass  # incomplete configuration, skip
        return backends

    @staticmethod
    def clear_cache() -> None:
        """Clear the backend cache. Use it to re-initialize after a configuration hot reload."""
        _BACKEND_CACHE.clear()

    @staticmethod
    def _is_backend_configured(name: str) -> bool:
        """Check whether a backend is configured in .env."""
        if name in _CLAUDE_NAMES:
            return any(get_env(k) is not None for k in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "ANTHROPIC_BASE_URL"))
        prefix = name.upper()
        return get_env(f"{prefix}_API_KEY") is not None or get_env(f"{prefix}_BASE_URL") is not None

    @staticmethod
    def _load_backend_config(name: str) -> dict:
        """Load the configuration dict of the given backend from environment variables.

        Environment variable naming: {PREFIX}_{PARAM}
          e.g. DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
        """
        if name in _CLAUDE_NAMES:
            return ModelRouter._load_claude_config()

        prefix = name.upper()

        api_key = get_env(f"{prefix}_API_KEY")
        base_url = get_env(f"{prefix}_BASE_URL")
        model = get_env(f"{prefix}_MODEL")

        if api_key is None and base_url is None:
            raise ValueError(
                f"Backend '{name}' is not configured. Please set "
                f"{prefix}_API_KEY and/or {prefix}_BASE_URL in .env or .env.local."
            )

        # build the parameter dict VLLMPolicy accepts
        config: dict = {}

        if model is not None:
            config["model_name"] = model
        if base_url is not None:
            config["base_url"] = base_url
        if api_key is not None:
            config["api_key"] = api_key

        # optional parameters
        temp = get_env(f"{prefix}_TEMPERATURE")
        if temp is not None:
            config["temperature"] = float(temp)

        max_tok = get_env(f"{prefix}_MAX_TOKENS")
        if max_tok is not None:
            config["max_tokens"] = int(max_tok)

        # provide a default for vllm (if the user did not configure a model)
        if name == "vllm" and "model_name" not in config:
            config["model_name"] = "Qwen/Qwen2.5-7B-Instruct"
        if name == "vllm" and "base_url" not in config:
            config["base_url"] = "http://localhost:8000/v1"
        if name == "vllm" and "api_key" not in config:
            config["api_key"] = "EMPTY"

        # provide defaults for deepseek
        if name == "deepseek" and "model_name" not in config:
            config["model_name"] = "deepseek-chat"
        if name == "deepseek" and "base_url" not in config:
            config["base_url"] = "https://api.deepseek.com/v1"

        # provide defaults for openai
        if name == "openai" and "model_name" not in config:
            config["model_name"] = "gpt-4o"
        if name == "openai" and "base_url" not in config:
            config["base_url"] = "https://api.openai.com/v1"

        # provide defaults for xiaomi mimo (official API, directly reachable in China)
        # official platform: https://platform.xiaomimimo.com/
        # to use the OpenRouter fallback channel, override manually in .env:
        #   MIMO_BASE_URL=https://openrouter.ai/api/v1
        #   MIMO_MODEL=xiaomi/mimo-v2.5-pro
        if name == "mimo" and "model_name" not in config:
            config["model_name"] = "mimo-v2.5-pro"
        if name == "mimo" and "base_url" not in config:
            config["base_url"] = "https://api.xiaomimimo.com/v1"

        return config

    @staticmethod
    def _load_claude_config() -> dict:
        """Claude backend configuration.

        Environment variables:
          ANTHROPIC_API_KEY  (or CLAUDE_API_KEY)  API key
          ANTHROPIC_BASE_URL (optional)            gateway / proxy address
          CLAUDE_MODEL       (optional)            default claude-sonnet-5-5
          CLAUDE_MAX_RETRIES / CLAUDE_TIMEOUT / CLAUDE_MAX_INPUT_CHARS (optional)
        """
        api_key = get_env("ANTHROPIC_API_KEY") or get_env("CLAUDE_API_KEY")
        base_url = get_env("ANTHROPIC_BASE_URL")
        if api_key is None and base_url is None:
            raise ValueError(
                "Backend 'claude' is not configured. Please set ANTHROPIC_API_KEY in .env or .env.local."
            )

        config: dict = {"model_name": get_env("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)}
        if api_key is not None:
            config["api_key"] = api_key
        if base_url is not None:
            config["base_url"] = base_url
        if get_env("CLAUDE_MAX_RETRIES") is not None:
            config["max_retries"] = int(get_env("CLAUDE_MAX_RETRIES"))
        if get_env("CLAUDE_TIMEOUT") is not None:
            config["timeout"] = float(get_env("CLAUDE_TIMEOUT"))
        if get_env("CLAUDE_MAX_INPUT_CHARS") is not None:
            config["max_input_chars"] = int(get_env("CLAUDE_MAX_INPUT_CHARS"))
        for suffix, cast, key in (("TEMPERATURE", float, "temperature"), ("MAX_TOKENS", int, "max_tokens")):
            val = get_env(f"CLAUDE_{suffix}")
            if val is not None:
                config[key] = cast(val)
        return config
