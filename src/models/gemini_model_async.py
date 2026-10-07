import asyncio
import io
import logging
import time
from datetime import datetime
from typing import List

from PIL import Image

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("Install google-genai: pip install google-genai")

from .base import BaseModel
from ..config.config import (
    GEMINI_MODEL_NAME, GEMINI_API_KEY, GEMINI_MAX_RETRIES, GEMINI_RETRY_DELAY
)
from .utils import _build_prompt, _parse_response

log = logging.getLogger(__name__)


def _pil_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class GeminiModelAsync(BaseModel):
    
    def __init__(
        self,
        model_name: str = GEMINI_MODEL_NAME,
        api_key: str = GEMINI_API_KEY,
        max_retries: int = GEMINI_MAX_RETRIES,
        retry_delay: float = GEMINI_RETRY_DELAY,
        max_concurrent: int = 10,
    ):
        self._model_name = model_name
        self._api_key = api_key
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._max_concurrent = max_concurrent  
        self._client = None
        self._semaphore = None
        self._concurrency_stats = {
            "peak_concurrent": 0,
            "total_calls": 0,
            "total_retries": 0,
        }

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
        log.info(f"[{self.name}] Client initialized (semaphore created per batch)")

    def predict_batch(self, images: List[Image.Image]) -> List[dict]:
       
        return asyncio.run(self._predict_batch_async(images))

    async def _predict_batch_async(self, images: List[Image.Image]) -> List[dict]:
      
        if not self._client:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Create semaphore in the current event loop context
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        
        tasks = [self._predict_single_async(img) for img in images]
        
        log.debug(f"[{self.name}] Starting {len(tasks)} concurrent predictions")
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                final_results.append({
                    "predicted": None,
                    "confidence": None,
                    "all_scores": {},
                    "raw_response": f"ASYNC_ERROR: {result}",
                })
            else:
                final_results.append(result)
        
        return final_results

    async def _predict_single_async(self, image: Image.Image) -> dict:
      
        async with self._semaphore:
            current_concurrent = self._max_concurrent - self._semaphore._value
            if current_concurrent > self._concurrency_stats["peak_concurrent"]:
                self._concurrency_stats["peak_concurrent"] = current_concurrent
            
            ts_start = datetime.now().isoformat()
            
            prompt = _build_prompt()
            img_bytes = _pil_to_jpeg_bytes(image)
            
            last_error = None
            for attempt in range(self._max_retries):
                try:
                    self._concurrency_stats["total_calls"] += 1
                    
                    # Run blocking API call in thread pool so concurrent requests can overlap
                    response = await asyncio.to_thread(
                        self._client.models.generate_content,
                        model=self._model_name,
                        contents=[
                            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                            prompt,
                        ],
                    )
                    
                    ts_end = datetime.now().isoformat()
                    log.debug(f"[{self.name}] API success | attempt={attempt+1} | "
                             f"start={ts_start} | end={ts_end}")
                    
                    raw = response.text
                    return {
                        "predicted": _parse_response(raw),
                        "confidence": None,
                        "all_scores": {},
                        "raw_response": raw,
                    }
                
                except Exception as exc:
                    last_error = exc
                    self._concurrency_stats["total_retries"] += 1
                    
                    if attempt < self._max_retries - 1:
                        wait_time = self._retry_delay * (2 ** attempt)
                        log.warning(f"[{self.name}] Retry {attempt+1}/{self._max_retries} "
                                  f"after {wait_time}s: {exc}")
                        await asyncio.sleep(wait_time)
            
            log.error(f"[{self.name}] All {self._max_retries} retries exhausted: {last_error}")
            return {
                "predicted": None,
                "confidence": None,
                "all_scores": {},
                "raw_response": f"API_ERROR_ALL_RETRIES: {last_error}",
            }

    def get_concurrency_stats(self) -> dict:
        return {
            **self._concurrency_stats,
            "max_concurrent_limit": self._max_concurrent,
        }
