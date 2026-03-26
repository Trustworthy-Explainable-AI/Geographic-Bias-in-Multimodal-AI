import io
import time
from typing import List, Optional

from PIL import Image
from google import genai
from google.genai import types

from .base import BaseModel
from ..config import (
    GEMINI_MODEL_NAME,
    GEMINI_API_KEY,
    GEMINI_MAX_RETRIES,
    GEMINI_RETRY_DELAY,
    CATEGORY_PROMPTS,
)


def _build_prompt(categories: List[str]) -> str:
    options = "\n".join(
        f"  {chr(65 + i)}) {CATEGORY_PROMPTS.get(cat, cat)}"
        for i, cat in enumerate(categories)
    )
    return (
        "You are a concise image classifier. "
        "Which of the following best describes the main subject of this image?\n"
        f"Options:\n{options}\n\n"
        "Reply with only the option letter (e.g. A, B, C …). "
        "Do not include any explanation."
    )


def _parse_response(raw: str, categories: List[str]) -> Optional[str]:
    clean = raw.strip().upper()
    if clean and clean[0].isalpha():
        idx = ord(clean[0]) - ord("A")
        if 0 <= idx < len(categories):
            return categories[idx]
    return None


def _pil_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class GeminiModel(BaseModel):

    def __init__(
        self,
        model_name:  str   = GEMINI_MODEL_NAME,
        api_key:     str   = GEMINI_API_KEY,
        max_retries: int   = GEMINI_MAX_RETRIES,
        retry_delay: float = GEMINI_RETRY_DELAY,
    ):
        self._model_name  = model_name
        self._api_key     = api_key
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._client      = None

    @property
    def name(self) -> str:
        return "gemini"

    def load(self) -> None:
        if self._client is not None:
            return
        if not self._api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set.\n"
                "Get a free key at https://aistudio.google.com/"
            )
        self._client = genai.Client(api_key=self._api_key)

    def predict_batch(self, images: List[Image.Image], categories: List[str]) -> List[dict]:
        return [self._predict_single(img, categories) for img in images]

    def _predict_single(self, image: Image.Image, categories: List[str]) -> dict:
        time.sleep(6)
        prompt    = _build_prompt(categories)
        img_bytes = _pil_to_jpeg_bytes(image)

        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        prompt,
                    ],
                )
                raw = response.text
                return {
                    "predicted":    _parse_response(raw, categories),
                    "confidence":   None,
                    "all_scores":   {},
                    "raw_response": raw,
                }
            except Exception as exc:
                last_error = exc
                time.sleep(self._retry_delay * (2 ** attempt))

        return {
            "predicted":    None,
            "confidence":   None,
            "all_scores":   {},
            "raw_response": f"API_ERROR: {last_error}",
        }
