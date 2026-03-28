from typing import List

from ..config import CATEGORY_PROMPTS, INFERENCE_PROMPT_TEMPLATE



def _build_prompt(categories: List[str]) -> str:
    options = "\n".join(
        f"  {chr(65 + i)}) {CATEGORY_PROMPTS.get(cat, cat)}"
        for i, cat in enumerate(categories)
    )
    return INFERENCE_PROMPT_TEMPLATE.format(options=options)
 

def _parse_response(raw: str, categories: List[str]):
    clean = raw.strip().upper()
    if clean and clean[0].isalpha():
        idx = ord(clean[0]) - ord("A")
        if 0 <= idx < len(categories):
            return categories[idx]
    return None
 