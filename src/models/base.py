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
        Download or initialise model weights or API client.

        Called once before the first call to predict_batch().
        """
        ...

    @abstractmethod
    def predict_batch(self, images: List[Image.Image]) -> List[dict]:
        """
        Run open-ended inference on a batch of PIL images.

        No candidate labels are provided — the model must produce a free-form
        label for the most prominent subject it can identify.

        Parameters
        images : List of PIL Image objects, already opened and in RGB mode.

        Returns
        A list of dicts, one per image, each with these keys:

            predicted    (str)           – free-form label, lowercased and
                                          whitespace-normalised; the literal
                                          string "abstention" if the model
                                          could not identify the subject.
            confidence   (float | None) – model confidence if available,
                                          else None.
            all_scores   (dict)         – always empty {} in open-ended mode.
            raw_response (str | None)   – raw text output; None for CLIP.

        Returning "abstention" for `predicted` signals the model could not
        identify the subject. This is preserved as-is in the output CSV.
        """
        ...

    def unload(self) -> None:
        """Release GPU memory and any other held resources."""
        pass