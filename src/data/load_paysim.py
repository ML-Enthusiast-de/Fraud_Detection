from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED = ["step","type","amount","nameOrig","nameDest","isFraud"]

def load_and_profile(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound: {list(df.columns)}")

    print("\n=== PaySim profile ===")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("Fraud rate:", float(df["isFraud"].mean()))
    print("Step range:", int(df["step"].min()), "..", int(df["step"].max()))
    print("\nTop types:\n", df["type"].value_counts().head(10).to_string())
    return df


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / "data" / "raw" / "paysim.parquet"

    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found.\nRun src/data/download_paysim.py first."
        )

    load_and_profile(data_path)
