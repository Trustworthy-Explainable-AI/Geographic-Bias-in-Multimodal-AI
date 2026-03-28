import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
 
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"

EVAL_SUBSET_PATH = DATA_DIR / "dollar_street_eval_subset_v2.csv"
IMAGE_ROOT = Path(
    os.environ.get("DOLLAR_STREET_IMAGE_ROOT", str(DATA_DIR / "images"))
)

CATEGORY_PROMPTS: dict[str, str] = {
    "toilet":                    "toilet or latrine",
    "tv":                        "television or TV screen",
    "stove/hob":                 "stove or cooking hob",
    "everyday shoes":            "everyday shoes, sandals or footwear",
    "washing clothes/cleaning":  "washing machine or clothes being washed",
    "drinking water":            "drinking water or water container",
    "cooking food":              "food being cooked or prepared",
    "clothes":                   "clothing or garments",
}

CATEGORIES: list[str] = list(CATEGORY_PROMPTS.keys())
CLIP_MODEL_NAME      = "ViT-B/32"
CLIP_PROMPT_TEMPLATE = "a photo of {label}"

QWEN_MODEL_NAME    = "Qwen/Qwen2-VL-7B-Instruct"
QWEN_MAX_NEW_TOKENS = 16  
QWEN_DEVICE = "auto"

GEMINI_MODEL_NAME  = "gemini-2.0-flash"
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY")
GEMINI_MAX_RETRIES = 2
GEMINI_RETRY_DELAY = 2 

INFERENCE_PROMPT_TEMPLATE = (
    "Which of the following best describes the main subject of this image?\n"
    "Options:\n{options}\n\n"
    "Reply with only the option letter (e.g. A, B, C …). "
    "Do not include any explanation."
) 

BATCH_SIZE  = 16