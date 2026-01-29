from __future__ import annotations

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score
from lightgbm import LGBMClassifier
import joblib


warnings.filterwarnings("ignore", message="X does not have valid feature names*")

# =========================
# CONFIG
# =========================
USE_BALANCE_ANOMALY_FEATURES = False  # <-- set True for "upper bound" model
ALLOWED_TYPES_GATE = {"TRANSFER", "CASH_OUT"}  # stage-0 gate used in reporting
# =========================


def make_ohe():
    """Compatible OneHotEncoder across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def time_split(df: pd.DataFrame, time_col: str = "step"):
    """Chronological 70/15/15 split."""
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    i1 = int(0.70 * n)
    i2 = int(0.85 * n)
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def operating_point_under_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float):
    """
    Rank-based operating point: choose max recall such that FPR <= target_fpr.
    Returns: (recall, precision, threshold, k_flagged)
    """
    order = np.argsort(-y_score)
    y = y_true[order]
    s = y_score[order]

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.0, float("inf"), 0

    is_pos = (y == 1)
    is_neg = ~is_pos
    cum_tp = np.cumsum(is_pos)
    cum_fp = np.cumsum(is_neg)

    fpr = cum_fp / n_neg
    recall = cum_tp / n_pos
    precision = cum_tp / np.maximum(1, (cum_tp + cum_fp))

    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0, 0.0, float("inf"), 0

    j = valid[np.argmax(recall[valid])]
    return float(recall[j]), float(precision[j]), float(s[j]), int(j + 1)


def topk_capture(y_true: np.ndarray, y_score: np.ndarray, top_frac: float) -> dict:
    n = len(y_true)
    k = max(1, int(round(top_frac * n)))
    idx = np.argsort(-y_score)[:k]

    fraud_in_top = int(y_true[idx].sum())
    total_fraud = int(y_true.sum())
    capture = fraud_in_top / total_fraud if total_fraud > 0 else 0.0
    precision = fraud_in_top / k if k > 0 else 0.0

    return {
        "top_frac": float(top_frac),
        "k": int(k),
        "fraud_in_top": int(fraud_in_top),
        "total_fraud": int(total_fraud),
        "capture_rate": float(capture),
        "precision_in_top": float(precision),
    }


def eval_operating_points(tag: str, y: np.ndarray, score: np.ndarray):
    fpr_targets = [0.001, 0.005, 0.01, 0.02, 0.05]
    rows = []
    for fpr_t in fpr_targets:
        rec, prec, thr, k = operating_point_under_fpr(y, score, fpr_t)
        rows.append({"target_fpr": fpr_t, "threshold": thr, "k_flagged": k, "recall": rec, "precision": prec})
    top_fracs = [0.001, 0.002, 0.005, 0.01, 0.02]
    topk = [topk_capture(y, score, f) for f in top_fracs]
    return {"tag": tag, "recall_at_fpr": rows, "topk_capture": topk}


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / "data" / "processed" / "paysim_with_temporal_graph_features.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run feature builder first.")

    out_dir = repo_root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path).sort_values("step").reset_index(drop=True)
    df["isFraud"] = df["isFraud"].astype(int)

    # Base + temporal/graph columns
    base_cols = [
        "step", "type", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
    ]

    graph_cols = [
        # 1h/24h/7d velocity
        "orig_out_cnt_1h", "orig_out_sum_1h",
        "orig_out_cnt_24h", "orig_out_sum_24h",
        "orig_out_cnt_7d", "orig_out_sum_7d",
        "orig_distinct_dest_7d",

        "dest_in_cnt_1h", "dest_in_sum_1h",
        "dest_in_cnt_24h", "dest_in_sum_24h",
        "dest_in_cnt_7d", "dest_in_sum_7d",
        "dest_distinct_orig_7d",

        # burstiness
        "orig_burst_cnt_1h_vs_7d",
        "dest_burst_cnt_1h_vs_7d",

        # pair + novelty
        "pair_cnt_7d",
        "is_new_counterparty_30d",

        # time since last
        "orig_time_since_last",
        "dest_time_since_last",
        "pair_time_since_last",
    ]

    # Optional balance anomaly features (often proxy-label-ish on synthetic PaySim)
    balance_cols = []
    if USE_BALANCE_ANOMALY_FEATURES:
        df["orig_delta"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
        df["dest_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]
        df["orig_error"] = (df["oldbalanceOrg"] - df["amount"]) - df["newbalanceOrig"]
        df["dest_error"] = (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]
        df["abs_orig_error"] = df["orig_error"].abs()
        df["abs_dest_error"] = df["dest_error"].abs()
        balance_cols = ["orig_delta", "dest_delta", "orig_error", "dest_error", "abs_orig_error", "abs_dest_error"]

    feature_cols = base_cols + graph_cols + balance_cols

    train_df, val_df, test_df = time_split(df, "step")
    X_train, y_train = train_df[feature_cols], train_df["isFraud"].values
    X_val, y_val = val_df[feature_cols], val_df["isFraud"].values
    X_test, y_test = test_df[feature_cols], test_df["isFraud"].values

    cat_cols = ["type"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", make_ohe(), cat_cols),
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

    # Probabilities for class 1
    fraud_col = list(pipe.named_steps["model"].classes_).index(1)
    val_score = pipe.predict_proba(X_val)[:, fraud_col]
    test_score = pipe.predict_proba(X_test)[:, fraud_col]

    # Basic AUC metrics
    metrics = {
        "use_balance_anomaly_features": bool(USE_BALANCE_ANOMALY_FEATURES),
        "rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "fraud_rate_total": float(df["isFraud"].mean()),
        "fraud_rate_train": float(y_train.mean()),
        "fraud_rate_test": float(y_test.mean()),
        "val_roc_auc": float(roc_auc_score(y_val, val_score)),
        "val_pr_auc": float(average_precision_score(y_val, val_score)),
        "test_roc_auc": float(roc_auc_score(y_test, test_score)),
        "test_pr_auc": float(average_precision_score(y_test, test_score)),
        "scale_pos_weight": float(spw),
        "n_features": int(len(feature_cols)),
        "feature_cols": feature_cols,
    }

    # RAW operating points (rank-based)
    metrics["operating_points_raw"] = eval_operating_points("RAW", y_test, test_score)

    # GATED operating points (set score=0 for non-allowed types)
    mask = test_df["type"].isin(ALLOWED_TYPES_GATE).values
    gated_score = test_score.copy()
    gated_score[~mask] = 0.0
    metrics["operating_points_gated"] = eval_operating_points(
        f"GATED({sorted(ALLOWED_TYPES_GATE)})", y_test, gated_score
    )

    # Save artifacts with distinct names
    suffix = "full" if USE_BALANCE_ANOMALY_FEATURES else "behavioral"
    model_path = out_dir / f"temporal_graph_{suffix}_lgbm.joblib"
    metrics_path = out_dir / f"temporal_graph_{suffix}_metrics.json"

    joblib.dump(pipe, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("\n=== Temporal+Graph Model Metrics ===")
    for k in ["val_roc_auc", "val_pr_auc", "test_roc_auc", "test_pr_auc", "n_features"]:
        print(f"{k}: {metrics[k]}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")

    # Print a quick gated summary table
    gated_rows = metrics["operating_points_gated"]["recall_at_fpr"]
    print("\nGATED recall@FPR:")
    for r in gated_rows:
        print(r)


if __name__ == "__main__":
    main()
