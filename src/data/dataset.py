from pathlib import Path
from typing import Optional
 
import pandas as pd
from PIL import Image
 
from ..config.config import EVAL_SUBSET_PATH, IMAGE_ROOT
 
 
def load_eval_subset(path: Path = EVAL_SUBSET_PATH, sample_size: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
    
    df = pd.read_csv(path)
    required = {"id", "imageRelPath", "region.id", "income_quintile", "topic_single"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Evaluation subset is missing expected columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )
 
    if sample_size is not None:
        df = df.sample(n=min(sample_size, len(df)), random_state=random_state)
        
    return df.reset_index(drop=True)
 
 
def resolve_image_path(rel_path: str, image_root: Path = IMAGE_ROOT) -> Path:
    
    return image_root / rel_path.lstrip("/")
 
 
def load_image(rel_path: str, image_root: Path = IMAGE_ROOT) -> Image.Image:
   
    full_path = resolve_image_path(rel_path, image_root)
    if not full_path.exists():
        raise FileNotFoundError(f"Image not found: {full_path}")
    return Image.open(full_path).convert("RGB")