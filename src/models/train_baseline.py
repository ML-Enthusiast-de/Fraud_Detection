from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
from lightgbm import LGBMClassifier
import joblib




def time_split(df: pd.DataFrame, time_col: str = "step"):
    """Chronological 70/15/15 split."""
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    i1 = int(0.70 * n)
    i2 = int(0.85 * n)
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """Compute recall when false positive rate <= target_fpr using ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0
    return float(np.max(tpr[idx]))


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / "data" / "raw" / "paysim.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run download first.")

    out_dir = repo_root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)

    # label
    y = df["isFraud"].astype(int)

    # features (baseline: mostly raw transaction fields)
    feature_cols = [
        "step", "type", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
    ]
    X = df[feature_cols].copy()

    train_df, val_df, test_df = time_split(df, "step")

    X_train, y_train = train_df[feature_cols], train_df["isFraud"].astype(int)
    X_val, y_val = val_df[feature_cols], val_df["isFraud"].astype(int)
    X_test, y_test = test_df[feature_cols], test_df["isFraud"].astype(int)

    cat_cols = ["type"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    # Handle imbalance: tell model positives are rare
    # scale_pos_weight ≈ (#neg / #pos) in train
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    spw = (n_neg / max(n_pos, 1))

    model = LGBMClassifier(
        verbose=-1,           # hides the spam
        min_child_samples=5,  # allow smaller leaves (helps with rare positives)
        min_split_gain=0.0,
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=96,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        scale_pos_weight=spw,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    proba_val = pipe.predict_proba(X_val)
    proba_test = pipe.predict_proba(X_test)

    model = pipe.named_steps["model"]
    print("Model classes_:", model.classes_)  
    
    fraud_col = list(model.classes_).index(1)

    val_score = proba_val[:, fraud_col]
    test_score = proba_test[:, fraud_col]


    metrics = {
        "rows": len(df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "fraud_rate_total": float(df["isFraud"].mean()),
        "fraud_rate_train": float(y_train.mean()),
        "fraud_rate_test": float(y_test.mean()),
        "val_roc_auc": float(roc_auc_score(y_val, val_score)),
        "val_pr_auc": float(average_precision_score(y_val, val_score)),
        "test_roc_auc": float(roc_auc_score(y_test, test_score)),
        "test_pr_auc": float(average_precision_score(y_test, test_score)),
        "test_recall_at_fpr_0_1pct": recall_at_fpr(y_test.values, test_score, 0.001),
        "test_recall_at_fpr_0_5pct": recall_at_fpr(y_test.values, test_score, 0.005),
        "scale_pos_weight": float(spw),
    }

    joblib.dump(pipe, out_dir / "baseline_lgbm.joblib")
    (out_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n=== Baseline metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
