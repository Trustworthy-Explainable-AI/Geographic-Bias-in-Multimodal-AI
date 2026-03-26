import argparse
import logging
import sys
from pathlib import Path
from typing import List
 
import pandas as pd
 
from ..config.config import (
    CATEGORIES,
    RESULTS_DIR,
    EVAL_SUBSET_PATH,
    IMAGE_ROOT,
    BATCH_SIZE,
)
from ..data.dataset import load_eval_subset, load_image
from ..evaluate import classify_error, tag_collapse, compute_summary
from ..models import CLIPModel, QwenVLModel, GeminiModel
from ..models.base import BaseModel
 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)
 

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "clip":   CLIPModel,
    "qwen":   QwenVLModel,
    "gemini": GeminiModel,
}
 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-model inference on the Dollar Street evaluation subset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_REGISTRY),
        default=["clip"],
        help="Models to run. Multiple allowed, e.g. --models clip qwen gemini.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N images. Useful for smoke-testing (e.g. --sample 30).",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=IMAGE_ROOT,
        help="Root directory containing Dollar Street images.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory where output CSVs are written.",
    )
    parser.add_argument(
        "--eval-subset",
        type=Path,
        default=EVAL_SUBSET_PATH,
        help="Path to dollar_street_eval_subset_v2.csv.",
    )
    return parser.parse_args()
 

def run_model(
    model:       BaseModel,
    df:          pd.DataFrame,
    image_root:  Path,
    categories:  List[str],
    batch_size:  int,
) -> pd.DataFrame:
    """_summary_

    Args:
        model (BaseModel): _description_
        df (pd.DataFrame): _description_
        image_root (Path): _description_
        categories (List[str]): _description_
        batch_size (int): _description_

    Returns:
        pd.DataFrame: _description_
    """
    records  = []
    skipped  = 0
    total    = len(df)
 
    log.info(f"[{model.name}] Starting inference on {total} images …")
 
    for batch_start in range(0, total, batch_size):
        batch_rows = df.iloc[batch_start : batch_start + batch_size]

        images:     list = []
        valid_rows: list = []
        for _, row in batch_rows.iterrows():
            try:
                img = load_image(row["imageRelPath"], image_root)
                images.append(img)
                valid_rows.append(row)
            except FileNotFoundError as exc:
                log.warning(f"  Skipped (not found): {exc}")
                skipped += 1
 
        if not images:
            continue
 
        predictions = model.predict_batch(images, categories)
 
        for row, pred in zip(valid_rows, predictions):
            ground_truth = row["topic_single"]
            correct      = (pred["predicted"] == ground_truth)
            error_type   = classify_error(correct, pred["predicted"], categories)
 
            records.append({
                "model":           model.name,
                "image_id":        row["id"],
                "region":          row["region.id"],
                "income_quintile": row["income_quintile"],
                "ground_truth":    ground_truth,
                "predicted":       pred["predicted"],
                "correct":         correct,
                "confidence":      pred["confidence"],
                "raw_response":    pred["raw_response"],
                "error_type":      error_type,
            })
 
        done = min(batch_start + batch_size, total)
        log.info(f"  [{model.name}] {done:>5}/{total} images processed …")
 
    result_df = pd.DataFrame(records)
 
    result_df = tag_collapse(result_df)
 
    n_correct = result_df["correct"].sum()
    n_total   = len(result_df)
    log.info(
        f"[{model.name}] Done. "
        f"Accuracy: {n_correct}/{n_total} = {n_correct/n_total:.1%}. "
        f"Skipped: {skipped} images."
    )
    return result_df
 

def save_results(
    result_df:   pd.DataFrame,
    results_dir: Path,
    model_name:  str,
) -> None:
    """_summary_

    Args:
        result_df (pd.DataFrame): _description_
        results_dir (Path): _description_
        model_name (str): _description_
    """
    results_dir.mkdir(parents=True, exist_ok=True)
 
    raw_path = results_dir / f"{model_name}_predictions.csv"
    result_df.to_csv(raw_path, index=False)
    log.info(f"  Saved predictions → {raw_path}")
 
    summaries = compute_summary(result_df)
    for table_name, table_df in summaries.items():
        path = results_dir / f"{model_name}_{table_name}.csv"
        table_df.to_csv(path, index=False)
        log.info(f"  Saved {table_name:15s} → {path}")
 

def main() -> None:
    args = parse_args()
    if not args.image_root.exists():
        log.warning(
            f"Image root does not exist: {args.image_root}\n"
            "  Set the correct path with --image-root or the env var "
            "DOLLAR_STREET_IMAGE_ROOT.\n"
            "  Images that cannot be found will be skipped during inference."
        )

    log.info(f"Loading eval subset: {args.eval_subset}")
    if not args.eval_subset.exists():
        log.error(
            f"Evaluation subset not found: {args.eval_subset}\n"
            "  Place dollar_street_eval_subset_v2.csv in the data/ directory."
        )
        sys.exit(1)
 
    df = load_eval_subset(path=args.eval_subset, sample_size=args.sample)
    log.info(
        f"  {len(df)} images | {df['region.id'].nunique()} regions | "
        f"{df['income_quintile'].nunique()} quintiles | "
        f"{df['topic_single'].nunique()} categories"
    )
 
    for model_key in args.models:
        log.info(f"\n{'=' * 60}")
        log.info(f"  MODEL: {model_key.upper()}")
        log.info(f"{'=' * 60}")
 
        model = MODEL_REGISTRY[model_key]()
        try:
            model.load()
            result_df = run_model(
                model      = model,
                df         = df,
                image_root = args.image_root,
                categories = CATEGORIES,
                batch_size = BATCH_SIZE,
            )
            save_results(result_df, args.results_dir, model.name)
        except Exception:
            log.exception(f"[{model_key}] Fatal error — skipping this model.")
        finally:
            model.unload()
 
    log.info("\nAll done. Results written to: %s", args.results_dir)
 
 
if __name__ == "__main__":
    main()