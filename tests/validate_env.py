#!/usr/bin/env python3
"""Verify that the .env / .env.local configuration is correct"""
from __future__ import annotations

import os
import sys

# Load .env
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
local_env = os.path.join(os.getcwd(), ".env.local")
if os.path.exists(local_env):
    load_dotenv(dotenv_path=local_env, override=True)


def check(key: str, expected_prefix: str = "", optional: bool = False) -> str:
    """Check an environment variable."""
    val = os.getenv(key)
    if not val:
        if optional:
            return f"  ⚪ {key}: not set (optional, default value used)"
        return f"  ❌ {key}: not set (required)"

    # Masked display
    if "key" in key.lower() or "token" in key.lower():
        display = val[:6] + "***" + val[-4:] if len(val) > 10 else "***"
    else:
        display = val

    status = "✅"
    if expected_prefix and not val.startswith(expected_prefix):
        status = "⚠️ "
        display += f" (should start with {expected_prefix})"

    return f"  {status} {key}: {display}"


print("=" * 60)
print("LLM backend configuration check")
print("=" * 60)
print(check("DEEPSEEK_API_KEY", optional=True))
print(check("DEEPSEEK_BASE_URL", expected_prefix="https://"))
print(check("MIMO_API_KEY", optional=True))
print(check("MIMO_BASE_URL", expected_prefix="https://"))
print(check("MIMO_MODEL"))

print()
print("=" * 60)
print("Tool layer configuration check")
print("=" * 60)
print(check("SERPAPI_KEY", optional=True))
print(check("SEARCH_BACKEND", optional=True))
print(check("BING_SEARCH_KEY", optional=True))
print(check("ARXIV_READER_BACKEND", optional=True))
print(check("SEMANTIC_SCHOLAR_API_KEY", optional=True))

print()
print("=" * 60)
print("LangSmith tracing configuration (optional)")
print("=" * 60)
print(check("LANGSMITH_TRACING", optional=True))
print(check("LANGSMITH_API_KEY", optional=True))

print()
print("=" * 60)
print("Key configuration suggestions")
print("=" * 60)

# Check mimo configuration
mimo_url = os.getenv("MIMO_BASE_URL", "")
mimo_model = os.getenv("MIMO_MODEL", "")
if "openrouter" in mimo_url and not mimo_model.startswith("xiaomi/"):
    print("  ⚠️  MIMO configuration mismatch:")
    print("      You are using the OpenRouter URL, but the model name has no xiaomi/ prefix")
    print("      Fixes:")
    print("        Option A (recommended): MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1")
    print("        Option B: MIMO_MODEL=xiaomi/mimo-v2.5-pro")
elif "xiaomimimo" in mimo_url and mimo_model.startswith("xiaomi/"):
    print("  ⚠️  MIMO configuration mismatch:")
    print("      You are using the official Xiaomi URL, but the model name has the xiaomi/ prefix (this is the OpenRouter format)")
    print("      Fix: MIMO_MODEL=mimo-v2.5-pro")
else:
    print("  ✅ MIMO configuration looks correct")

# Check search backend
search_backend = os.getenv("SEARCH_BACKEND", "serpapi").lower()
if search_backend == "serpapi" and not os.getenv("SERPAPI_KEY"):
    print("  ⚠️  SEARCH_BACKEND=serpapi，but SERPAPI_KEY is not set")
elif search_backend == "bing" and not os.getenv("BING_SEARCH_KEY"):
    print("  ⚠️  SEARCH_BACKEND=bing，but BING_SEARCH_KEY is not set")
else:
    print("  ✅ Search backend configuration is correct")

print()
