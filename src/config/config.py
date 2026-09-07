import os
from pathlib import Path
from dotenv import load_dotenv
 
load_dotenv()
 
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "result"
 
EVAL_SUBSET_PATH = DATA_DIR / "dollar_street_eval_subset_4000.csv"
IMAGE_ROOT = Path(
    os.environ.get("DOLLAR_STREET_IMAGE_ROOT", str(DATA_DIR / "images"))
)
 
OPEN_ENDED_PROMPT = (
    "Look at this image carefully.\n"
    "Categorize the image based on what you can see.\n"
    "Be concise; give a short label only (e.g. 'wooden chair', 'gas stove', 'flip-flops').\n"
    "Do not Hallucinate. If you genuinely cannot identify the subject, reply exactly with: 'Unknown'.\n"
    "Do not go against the above instructions for every image"
)

SEM_SIM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # cosine similarity
CTX_SIM_MODEL_NAME = "cross-encoder/stsb-roberta-base"        #contextual similarity
 

SEM_SIM_THRESHOLD  = 0.45   # cosine similarity in
CTX_SIM_THRESHOLD  = 0.45   # sigmoid-normalised cross-encoder score in
 
CLIP_MODEL_NAME    = "ViT-B/32"
 
QWEN_MODEL_NAME    = "Qwen/Qwen2-VL-7B-Instruct"
QWEN_MAX_NEW_TOKENS = 16
QWEN_DEVICE        = "auto"
 
GEMINI_MODEL_NAME  = "gemini-2.5-flash"
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY")
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_DELAY = 5

# Batch size: 32 is safe on L4 (23.7GB), reduces runtime ~50%
# BATCH_SIZE  = 32 # for qwen, 
BATCH_SIZE  = 40 # for gemini, make multiple of 10 to maximise concurrency without leaving too many stragglers at the end