from abc import ABC, abstractmethod
from typing import List
 
from PIL import Image
 
 
class BaseModel(ABC):
    """
    Minimal interface shared by CLIP, Qwen-VL, and Gemini.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Short identifier for this model, used in output filenames.
        Examples: "clip_vitb32", "qwen_vl", "gemini"
        """
        ...
 
    @abstractmethod
    def load(self) -> None:
        """
        Download or initialises model weights or API client.
 
        Called once before the first call to predict_batch().
        """
        ...
 
    @abstractmethod
    def predict_batch(
        self,
        images: List[Image.Image],
        categories: List[str],
    ) -> List[dict]:
        """
        Run inference on a batch of PIL images.
 
        Parameters:
        images     : List of PIL Image objects, already opened and in RGB mode.
        categories : The candidate category labels for this run.
 
        Returns:
        A list of dicts, one per image, each with these keys:
 
            predicted   (str | None)   – winning category, or None if the
                                         model abstained / returned garbage
            confidence  (float | None) – score for the predicted category;
                                         None if the model does not expose it
            all_scores  (dict)         – {category: score} for every label;
                                         empty dict for generative models
            raw_response (str | None)  – raw text output for VQA models;
                                         None for CLIP
 
        Returning None for `predicted` signals an abstention.
        Returning a string that is NOT in `categories` signals a hallucination.
        Both are handled automatically in evaluate.py.
        """
        ...
 
    def unload(self) -> None:
        """
        Release GPU memory and any other held resources.
        """
        pass