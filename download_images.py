import shutil
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
import kagglehub

DATASET   = "mlcommons/the-dollar-street-dataset"
CSV_PATH  = "data/dollar_street_eval_subset_v2.csv"
IMAGE_DIR = Path("data/images")

df = pd.read_csv(CSV_PATH)
image_paths = df["imageRelPath"].tolist()

dataset_root = Path(kagglehub.dataset_download(DATASET)) / "dataset_dollarstreet"
print(f"Cached at: {dataset_root}")

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
missing = []

for i, rel_path in enumerate(image_paths):
    src  = dataset_root / rel_path
    dest = IMAGE_DIR / rel_path
    if dest.exists():
        continue
    if not src.exists():
        missing.append(rel_path)
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(image_paths)} copied")

print(f"Done. {len(image_paths) - len(missing)} copied, {len(missing)} missing.")

