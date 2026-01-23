from __future__ import annotations

from pathlib import Path
import pandas as pd

# This uses Hugging Face Datasets. It will download from the dataset hub and cache locally.
# Dataset page (PaySim-like): https://huggingface.co/datasets/purulalwani/Synthetic-Financial-Datasets-For-Fraud-Detection
DATASET_ID = "purulalwani/Synthetic-Financial-Datasets-For-Fraud-Detection"
SPLIT = "train"

def download_paysim(out_path: Path, sample_rows: int = 300_000, seed: int = 42) -> Path:
    """
    Downloads a PaySim-like dataset from Hugging Face and saves it as Parquet.

    Where does it download from?
    - From the Hugging Face dataset hub (DATASET_ID).
    - Files are cached by the `datasets` library (typically under ~/.cache/huggingface/datasets).
    """
    from datasets import load_dataset  # import here so error message is clearer if not installed

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET_ID, split=SPLIT)

    # Keep it fast: sample a subset for development (you can increase later)
    if sample_rows and sample_rows < len(ds):
        ds = ds.shuffle(seed=seed).select(range(sample_rows))

    df = ds.to_pandas()
    df.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    # Run-friendly defaults (no CLI needed)
    repo_root = Path(__file__).resolve().parents[2]  # .../repo/src/data/download_paysim.py -> repo
    out = repo_root / "data" / "raw" / "paysim.parquet"

    print("Downloading PaySim-like dataset from Hugging Face...")
    saved = download_paysim(out_path=out, sample_rows=300_000)
    print(f"Saved to: {saved}")
