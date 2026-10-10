"""Model presets and the model pairs each condition uses."""

from __future__ import annotations

from .sim import ModelCfg

PRESETS: dict[str, ModelCfg] = {
    # fast models: reasoning off
    "haiku": ModelCfg("anthropic/claude-haiku-5.5", {"enabled": False}, 400),
    "luna": ModelCfg("openai/gpt-6-luna", {"effort": "none"}, 400),
    # the same fast models allowed to think, for the single-fast baselines that do the tool work themselves
    "haiku-low": ModelCfg("anthropic/claude-haiku-5.5", {"effort": "low"}, 4000),
    "luna-low": ModelCfg("openai/gpt-6-luna", {"effort": "low"}, 4000),
    # slow models: reasoning is mandatory; low is their minimum
    "sonnet-low": ModelCfg("anthropic/claude-sonnet-5.5", {"effort": "low"}, 6000),
    "sonnet-high": ModelCfg("anthropic/claude-sonnet-5.5", {"effort": "high"}, 16000),
    "sol-low": ModelCfg("openai/gpt-6.1-sol", {"effort": "low"}, 6000),
    # the February pair, as the code at 95999dee5 ran it
    "gpt5mini-minimal": ModelCfg("openai/gpt-5-mini", {"effort": "minimal"}, 2000),
    "opus46-low": ModelCfg("anthropic/claude-opus-4.6", {"effort": "low"}, 6000),
    # the summer pair, as the code at 27e50e28f (11 July 2026) ran it
    "gpt54mini-low": ModelCfg("openai/gpt-5.4-mini", {"effort": "low"}, 4000),
    "terra-high": ModelCfg("openai/gpt-5.6-terra", {"effort": "high"}, 16000),
    # today's frontier slow model, at its minimum effort as Sonnet 5.5 is in today's tier
    "opus55-low": ModelCfg("anthropic/claude-opus-5.5", {"effort": "low"}, 6000),
    # the top slow tier of Plan amendment 4
    "opus55-high": ModelCfg("anthropic/claude-opus-5.5", {"effort": "high"}, 16000),
    # judge: a third family
    "judge": ModelCfg("google/gemini-3.8-flash", {"effort": "medium"}, 8000),
}

PAIRS: dict[str, tuple[str, str, str]] = {  # fast, slow, fast allowed to think
    "anthropic": ("haiku", "sonnet-low", "haiku-low"),
    "openai": ("luna", "sol-low", "luna-low"),
    "anthropic-high": ("haiku", "sonnet-high", "haiku-low"),
    # era pairs (Plan amendment 3): the fast model runs at its era setting, so there is no separate "think" variant
    "feb": ("gpt5mini-minimal", "opus46-low", "gpt5mini-minimal"),
    "summer": ("gpt54mini-low", "terra-high", "gpt54mini-low"),
    "frontier": ("haiku", "opus55-low", "haiku-low"),
    # Plan amendment 4: today's fast tier with three slow tiers ("anthropic" and "openai" above are the mid tier)
    "cheap-anthropic": ("haiku", "haiku", "haiku-low"),
    "cheap-openai": ("luna", "luna", "luna-low"),
    "top-anthropic": ("haiku", "opus55-high", "haiku-low"),
}
