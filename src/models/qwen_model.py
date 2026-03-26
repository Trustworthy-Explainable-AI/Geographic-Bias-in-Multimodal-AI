from typing import List
 
import torch
from PIL import Image

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from .base import BaseModel
from ..config import QWEN_MODEL_NAME, QWEN_MAX_NEW_TOKENS, QWEN_DEVICE, CATEGORY_PROMPTS
 

def _build_prompt(categories: List[str]) -> str:
    """_summary_

    Args:
        categories (List[str]): _description_

    Returns:
        str: _description_
    """
    options = "\n".join(
        f"  {chr(65 + i)}) {CATEGORY_PROMPTS.get(cat, cat)}"
        for i, cat in enumerate(categories)
    )
    return (
        "Which of the following best describes the main subject of this image?\n"
        f"Options:\n{options}\n\n"
        "Reply with only the option letter (e.g. A, B, C …). "
        "Do not include any explanation."
    )
 
 
def _parse_response(raw: str, categories: List[str]):
    """_summary_

    Args:
        raw (str): _description_
        categories (List[str]): _description_

    Returns:
        _type_: _description_
    """
    clean = raw.strip().upper()
    if clean and clean[0].isalpha():
        idx = ord(clean[0]) - ord("A")
        if 0 <= idx < len(categories):
            return categories[idx]
    return None
 

class QwenVLModel(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): _description_
    """
 
    def __init__(
        self,
        model_name:      str = QWEN_MODEL_NAME,
        device:          str = QWEN_DEVICE,
        max_new_tokens:  int = QWEN_MAX_NEW_TOKENS,
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
 
    def predict_batch(
        self,
        images: List[Image.Image],
        categories: List[str],
    ) -> List[dict]:
        """_summary_

        Args:
            images (List[Image.Image]): _description_
            categories (List[str]): _description_

        Returns:
            List[dict]: _description_
        """
        return [self._predict_single(img, categories) for img in images]
 
    def unload(self) -> None:
        """Release GPU memory."""
        del self._model, self._processor
        self._model = self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
    def _predict_single(self, image: Image.Image, categories: List[str]) -> dict:
        """_summary_

        Args:
            image (Image.Image): _description_
            categories (List[str]): _description_

        Returns:
            dict: _description_
        """
        prompt = _build_prompt(categories)
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
            "predicted":    _parse_response(raw, categories),
            "confidence":   None,
            "all_scores":   {},
            "raw_response": raw,
        }