from typing import Optional, List

from ..config.config import OPEN_ENDED_PROMPT

_ABSTENTION_TOKENS: frozenset[str] = frozenset({
    "unknown", "Unknown", "unsure", "unclear", "i don't know", "i cannot identify",
    "cannot identify", "not sure", "n/a", "none", "i'm not sure",
    "i am not sure", "cannot determine", "indeterminate",
})

def _build_prompt() -> str:
    """Return the open-ended, option-free inference prompt."""
    return OPEN_ENDED_PROMPT


def _parse_response(raw: str) -> str:
    """
    Extract a free-form label from a model's raw text output.
 
    Returns
    -------
    str : normalised label if identifiable,
          or the literal string "abstention" if the model said it doesn't know
          or returned empty output.
    """
    clean = raw.strip().strip(".,!?\"'").lower()
    if not clean or clean in _ABSTENTION_TOKENS:
        return "abstention"
    return " ".join(clean.split())
