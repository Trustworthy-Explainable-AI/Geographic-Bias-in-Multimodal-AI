from typing import List

import clip
import torch
from PIL import Image
import logging

from .base import BaseModel
from ..config.config import CLIP_MODEL_NAME

log = logging.getLogger(__name__)

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
        """
        CLIP override: Computes similarity against dynamic categories.
        """
        if not categories:
            log.error("CLIP requires categories for prediction. Returning abstentions.")
            return [{"predicted": "abstention", "confidence": 0.0, "all_scores": {}, "raw_response": None} for _ in images]

        # We wrap categories in a template to help CLIP (optional, but recommended)
        text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in categories]).to(self._device)
        
        with torch.no_grad():
            text_features = self._model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        results = []
        for img in images:
            img_tensor = self._preprocess(img).unsqueeze(0).to(self._device)
            with torch.no_grad():
                image_features = self._model.encode_image(img_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # Calculate Similarity
            # Logits are scaled by 100 per CLIP default implementation
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity.topk(1)
            
            best_idx = int(indices.item())
            conf_score = float(values.item())

            results.append({
                "predicted":    categories[best_idx],
                "confidence":   conf_score,
                "all_scores":   {},
                "raw_response": None
            })
 
        return results


    def unload(self) -> None:
        del self._model, self._preprocess
        self._model = self._preprocess = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()