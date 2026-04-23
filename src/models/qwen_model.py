from typing import List

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from .base import BaseModel
from ..config.config import QWEN_MODEL_NAME, QWEN_MAX_NEW_TOKENS, QWEN_DEVICE
from .utils import _build_prompt, _parse_response


class QwenVLModel(BaseModel):

    def __init__(
        self,
        model_name:     str = QWEN_MODEL_NAME,
        device:         str = QWEN_DEVICE,
        max_new_tokens: int = QWEN_MAX_NEW_TOKENS,
    ):
        self._model_name     = model_name
        self._device         = device
        self._max_new_tokens = max_new_tokens
        self._model          = None
        self._processor      = None

    @property
    def name(self) -> str:
        return "qwen_vl"

    def load(self) -> None:
        """Download and initialise Qwen2-VL weights. Idempotent."""
        if self._model is not None:
            return
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self._model_name,
            torch_dtype=dtype,
            device_map=self._device,
        )
        self._processor = AutoProcessor.from_pretrained(self._model_name)

    def predict_batch(self, images: List[Image.Image]) -> List[dict]:
        return [self._predict_single(img) for img in images]

    def unload(self) -> None:
        """Release GPU memory."""
        del self._model, self._processor
        self._model = self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _predict_single(self, image: Image.Image) -> dict:
        prompt   = _build_prompt()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text":  prompt},
                ],
            }
        ]

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_input],
            images=[image],
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        raw = self._processor.decode(new_tokens, skip_special_tokens=True)

        return {
            "predicted":    _parse_response(raw),
            "confidence":   None,
            "all_scores":   {},
            "raw_response": raw,
        }