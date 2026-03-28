from typing import List

import clip
import torch
from PIL import Image
 
from .base import BaseModel
from ..config.config import CLIP_MODEL_NAME, CLIP_PROMPT_TEMPLATE, CATEGORY_PROMPTS
 
 
class CLIPModel(BaseModel):

    def __init__(self, model_name: str = CLIP_MODEL_NAME):
        self._model_name = model_name
        self._model      = None
        self._preprocess = None
        self._device     = "cuda" if torch.cuda.is_available() else "cpu"
 
    @property
    def name(self) -> str:
        slug = self._model_name.lower().replace("/", "").replace("-", "")
        return f"clip_{slug}"
 
    def load(self) -> None:
        if self._model is not None:
            return
        self._model, self._preprocess = clip.load(self._model_name, device=self._device)
        self._model.eval()
 
    def predict_batch(self, images: List[Image.Image], categories: List[str]) -> List[dict]:
        
        text_features = self._encode_texts(categories)
        results = []
        for img in images:
            img_tensor   = self._preprocess(img).unsqueeze(0).to(self._device)
            img_features = self._encode_image(img_tensor)

            sims = (img_features @ text_features.T).squeeze(0)
            sims_cpu = sims.cpu().float().numpy()
 
            best_idx = int(sims_cpu.argmax())
            results.append({
                "predicted":    categories[best_idx],
                "confidence":   float(sims_cpu[best_idx]),
                "all_scores":   {cat: float(s) for cat, s in zip(categories, sims_cpu)},
                "raw_response": None
            })
 
        return results
 
    def unload(self) -> None:
        del self._model, self._preprocess
        self._model = self._preprocess = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
 
    def _encode_texts(self, categories: List[str]) -> torch.Tensor:
        prompts = [
            CLIP_PROMPT_TEMPLATE.format(label=CATEGORY_PROMPTS.get(cat, cat))
            for cat in categories
        ]
        tokens = clip.tokenize(prompts).to(self._device)
        with torch.no_grad():
            feats = self._model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats


    def _encode_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self._model.encode_image(img_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats