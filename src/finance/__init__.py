"""财报 / 金融文档研究场景：XBRL 取数、证据账本、财报专用 Agent。

保持本包 import 轻量（不在顶层引入 aiohttp / anthropic 等重依赖）。
"""
from __future__ import annotations

__all__ = ["xbrl", "evidence"]
