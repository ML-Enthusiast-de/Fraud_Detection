from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED = ["step","type","amount","nameOrig","nameDest","isFraud"]

def main():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "data" / "raw" / "paysim.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the download first.")

    df = pd.read_parquet(path)

    print("\n=== PaySim Profile ===")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Step range:", int(df["step"].min()), "..", int(df["step"].max()))
    print("Fraud rate:", float(df["isFraud"].mean()))
    print("\nTransaction types:\n", df["type"].value_counts().to_string())

    # sanity checks
    print("\n=== Sanity checks ===")
    print("Negative amounts:", int((df["amount"] < 0).sum()))
    print("Nulls per column:\n", df.isna().sum().sort_values(ascending=False).head(10).to_string())

    # fraud by type
    fraud_by_type = df.groupby("type")["isFraud"].mean().sort_values(ascending=False)
    print("\nFraud rate by type:\n", fraud_by_type.to_string())

if __name__ == "__main__":
    main()
