from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from lightgbm import LGBMClassifier
import joblib


def time_split(df: pd.DataFrame, time_col: str = "step"):
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    i1 = int(0.70 * n)
    i2 = int(0.85 * n)
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0
    return float(np.max(tpr[idx]))


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / "data" / "processed" / "paysim_with_temporal_graph_features.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run feature builder first.")

    out_dir = repo_root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)
    df["isFraud"] = df["isFraud"].astype(int)

    base_cols = [
        "step", "type", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
    ]
    graph_cols = [
        "orig_out_cnt_24h", "orig_out_sum_24h",
        "orig_out_cnt_7d", "orig_out_sum_7d",
        "orig_distinct_dest_7d",
        "dest_in_cnt_24h", "dest_in_sum_24h",
        "dest_in_cnt_7d", "dest_in_sum_7d",
        "dest_distinct_orig_7d",
        "pair_cnt_7d",
        "is_new_counterparty_30d",
    ]
    feature_cols = base_cols + graph_cols

    train_df, val_df, test_df = time_split(df, "step")
    X_train, y_train = train_df[feature_cols], train_df["isFraud"]
    X_val, y_val = val_df[feature_cols], val_df["isFraud"]
    X_test, y_test = test_df[feature_cols], test_df["isFraud"]

    cat_cols = ["type"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    spw = (n_neg / max(n_pos, 1))

    model = LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=128,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        scale_pos_weight=spw,
        verbose=-1,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    proba_val = pipe.predict_proba(X_val)
    proba_test = pipe.predict_proba(X_test)
    fraud_col = list(pipe.named_steps["model"].classes_).index(1)

    val_score = proba_val[:, fraud_col]
    test_score = proba_test[:, fraud_col]

    metrics = {
        "val_roc_auc": float(roc_auc_score(y_val, val_score)),
        "val_pr_auc": float(average_precision_score(y_val, val_score)),
        "test_roc_auc": float(roc_auc_score(y_test, test_score)),
        "test_pr_auc": float(average_precision_score(y_test, test_score)),
        "test_recall_at_fpr_0_1pct": recall_at_fpr(y_test.values, test_score, 0.001),
        "test_recall_at_fpr_0_5pct": recall_at_fpr(y_test.values, test_score, 0.005),
        "scale_pos_weight": float(spw),
        "n_features": len(feature_cols),
    }

    joblib.dump(pipe, out_dir / "temporal_graph_lgbm.joblib")
    (out_dir / "temporal_graph_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n=== Temporal+Graph feature model metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
