"""Shared helper utilities for LangGraph agents."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Union


SignalsInput = Optional[Union[Dict[str, Sequence[str]], Sequence[str], str]]


def normalize_signals(raw: SignalsInput) -> Optional[Dict[str, Sequence[str]]]:
    """Coerce arbitrary signal formats into a dict of string lists."""
    if raw is None:
        return None

    if isinstance(raw, dict):
        normalized: Dict[str, Sequence[str]] = {}
        for key, value in raw.items():
            if isinstance(value, str):
                normalized[key] = [value]
            elif isinstance(value, bool):
                normalized[key] = ["true" if value else "false"]
            elif isinstance(value, Iterable):
                normalized[key] = [str(item) for item in value]
            else:
                normalized[key] = [str(value)]
        return normalized

    if isinstance(raw, str):
        return {"notes": [raw]}

    return {"evidence": [str(item) for item in raw]}
