import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

from .config.config import RESULTS_DIR, EVAL_SUBSET_PATH, IMAGE_ROOT, BATCH_SIZE
from .data.dataset import load_eval_subset, load_image
from .evaluation.evaluate import SimilarityEvaluator, classify_error, compute_summary
from .models import CLIPModel, QwenVLModel, GeminiModel, GeminiModelAsync
from .models.base import BaseModel


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
    "gemini": GeminiModelAsync,  # Use the async version of Gemini
}


def get_checkpoint_path(results_dir: Path, model_name: str) -> Path:
    return results_dir / ".checkpoints" / f"{model_name}_checkpoint.csv"


def load_checkpoint(checkpoint_path: Path) -> pd.DataFrame | None:
    if checkpoint_path.exists():
        try:
            return pd.read_csv(checkpoint_path)
        except Exception as e:
            log.warning(f"Failed to load checkpoint: {e}")
            return None
    return None


def save_checkpoint(result_df: pd.DataFrame, checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(checkpoint_path, index=False)


def get_processed_image_ids(checkpoint_df: pd.DataFrame) -> set:
    return set(checkpoint_df["image_id"].tolist()) if checkpoint_df is not None else set()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run open-ended multi-model inference on the Dollar Street evaluation subset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_REGISTRY),
        default=["gemini"],
        help="Models to run. Multiple allowed, e.g. --models qwen gemini.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N images (e.g. --sample 30).",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available. Skips already-processed images.",
    )
    parser.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="Clear any existing checkpoint before starting.",
    )
 
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=1,
        metavar="N",
        help="Save checkpoint every N batches (default: 1). Use --checkpoint-freq 5 to save every 5 batches (~2.5% faster).",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpointing entirely. Fastest but no resume capability.",
    )
    return parser.parse_args()


def run_model(
    model:                 BaseModel,
    df:                    pd.DataFrame,
    image_root:            Path,
    evaluator:             SimilarityEvaluator,
    batch_size:            int,
    checkpoint_path:       Path | None = None,
    checkpoint_freq:       int = 1,
    resume:                bool = False,
    evaluate:              bool = True,
) -> pd.DataFrame:
    records = []
    skipped = 0
    total   = len(df)

    is_clip = "clip" in model.name
    unique_categories = []
    if is_clip:
        unique_categories = sorted(df["topic_single"].unique().tolist())
        log.info(f"[{model.name}] CLIP detected. Extracted {len(unique_categories)} unique categories from 'topic_single'.")

    processed_ids = set()
    if resume and checkpoint_path:
        checkpoint_df = load_checkpoint(checkpoint_path)
        if checkpoint_df is not None:
            processed_ids = get_processed_image_ids(checkpoint_df)
            records = checkpoint_df.to_dict('records')
            log.info(f"[{model.name}] ✓ Resuming from checkpoint: {len(processed_ids)} images already processed")
    
    if processed_ids:
        df_to_process = df[~df["id"].isin(processed_ids)].reset_index(drop=True)
        log.info(f"[{model.name}] Skipping {len(df) - len(df_to_process)} processed images. {len(df_to_process)} remaining.")
    else:
        df_to_process = df.reset_index(drop=True)

    log.info(f"[{model.name}] Starting open-ended inference on {len(df_to_process)} images …")

    batch_counter = 0
    for batch_start in range(0, len(df_to_process), batch_size):
        batch_rows = df_to_process.iloc[batch_start : batch_start + batch_size]
        batch_counter += 1
        batch_num = batch_start // batch_size + 1
        total_batches = (len(df_to_process) + batch_size - 1) // batch_size
        
        log.info(f"  [{batch_num}/{total_batches}] Processing batch ({batch_start + 1}–{min(batch_start + batch_size, len(df_to_process))} of {len(df_to_process)} images)...")

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
            except OSError as exc:  
                log.warning(f"  Skipped (corrupted): {exc}")
                skipped += 1
            except Exception as exc:
                 log.warning(f"  Skipped (unexpected error): {exc}")
                 skipped += 1

        if not images:
            continue

        # CLIP gets the dynamic categories; Qwen/Gemini use the default signature
        if is_clip:
            predictions = model.predict_batch(images, categories=unique_categories)
        else:
            predictions = model.predict_batch(images)

        for row, pred in zip(valid_rows, predictions):
            ground_truth = row["topic_single"]
            error_type, sem_sim, ctx_sim = classify_error(
                ground_truth=ground_truth,
                predicted=pred["predicted"],
                evaluator=evaluator,
            )

            records.append({
                "model":           model.name,
                "image_id":        row["id"],
                "region":          row["region.id"],
                "income_quintile": row["income_quintile"],
                "ground_truth":    ground_truth,
                "predicted":       pred["predicted"],
                "error_type":      error_type,
                "sem_similarity":  round(sem_sim, 4),
                "ctx_similarity":  round(ctx_sim, 4),
                "confidence":      pred["confidence"],
                "raw_response":    pred["raw_response"],
            })
        
        if checkpoint_path and batch_counter % checkpoint_freq == 0:
            result_df = pd.DataFrame(records)
            save_checkpoint(result_df, checkpoint_path)
            log.debug(f"    ✓ Checkpoint saved at batch {batch_counter} ({len(records)} total records)")
        
        log.info(f"  ✓ Batch {batch_num} complete ({len(records)}/{len(df_to_process) + skipped} records so far)")

    return pd.DataFrame(records)

def save_results(result_df: pd.DataFrame, results_dir: Path, model_name: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = results_dir / f"{model_name}_predictions.csv"
    result_df.to_csv(raw_path, index=False)
    log.info(f"  Saved predictions      → {raw_path}")

    summaries = compute_summary(result_df)
    for table_name, table_df in summaries.items():
        path = results_dir / f"{model_name}_{table_name}.csv"
        table_df.to_csv(path, index=False)
        log.info(f"  Saved {table_name:20s} → {path}")


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

    log.info("Loading similarity evaluator …")
    evaluator = SimilarityEvaluator()
    evaluator.load()
    log.info("  Similarity evaluator ready.")

    for model_key in args.models:
        log.info(f"\n{'=' * 60}")
        log.info(f"  MODEL: {model_key.upper()}")
        log.info(f"{'=' * 60}")

        model = MODEL_REGISTRY[model_key]()
        try:
            model.load()
            
         
            checkpoint_path = get_checkpoint_path(args.results_dir, model.name)
            if args.clear_checkpoint and checkpoint_path.exists():
                checkpoint_path.unlink()
                log.info(f"  Cleared checkpoint: {checkpoint_path}")
            
            result_df = run_model(
                model=model,
                df=df,
                image_root=args.image_root,
                evaluator=evaluator,
                batch_size=BATCH_SIZE,
                checkpoint_path=checkpoint_path,
                checkpoint_freq=args.checkpoint_freq,
                resume=args.resume,
                evaluate=True,
            )
            save_results(result_df, args.results_dir, model.name)
            
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                log.info(f"  ✓ Checkpoint removed after completion")
        except Exception:
            log.exception(f"[{model_key}] Fatal error — skipping this model.")
        finally:
            model.unload()

    log.info("\nAll done. Results written to: %s", args.results_dir)


if __name__ == "__main__":
    main()