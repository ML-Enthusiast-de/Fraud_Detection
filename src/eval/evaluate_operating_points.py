from __future__ import annotations

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", message="X does not have valid feature names*")


def load_scores(repo_root: Path):
    """
    Loads the processed dataset and the saved temporal-graph LightGBM model,
    then scores the TEST split (chronological 70/15/15).
    """
    import joblib

    data_path = repo_root / "data" / "processed" / "paysim_with_temporal_graph_features.parquet"
    model_path = repo_root / "artifacts" / "temporal_graph_lgbm.joblib"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run feature builder first.")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}. Train temporal graph model first.")

    df = pd.read_parquet(data_path).sort_values("step").reset_index(drop=True)

    # same split logic as training
    n = len(df)
    i1 = int(0.70 * n)
    i2 = int(0.85 * n)
    test_df = df.iloc[i2:].copy()

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

    X_test = test_df[feature_cols]
    y_test = test_df["isFraud"].astype(int).values

    pipe = joblib.load(model_path)
    proba = pipe.predict_proba(X_test)
    fraud_col = list(pipe.named_steps["model"].classes_).index(1)
    s_test = proba[:, fraud_col]

    return test_df, y_test, s_test


def operating_point_under_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float):
    """
    Rank-based operating point:
    - Sort by score desc
    - Sweep threshold down and track cumulative FP/TP
    - Choose the point with maximum recall while FPR <= target_fpr

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
    thr = float(s[j])
    k = int(j + 1)

    return float(recall[j]), float(precision[j]), thr, k


def topk_capture(y_true: np.ndarray, y_score: np.ndarray, top_frac: float) -> dict:
    """
    Look at the top top_frac fraction of transactions by score.
    Returns capture rate and precision within that band.
    """
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


def main():
    repo_root = Path(__file__).resolve().parents[2]

    test_df, y_test, s_test = load_scores(repo_root)

    print("\n=== Operating Point Evaluation (TEST set) ===")
    print("Test rows:", len(y_test))
    print("Test fraud count:", int(y_test.sum()))
    print("Test fraud rate:", float(y_test.mean()))

    # Recall at various FPR budgets
    fpr_targets = [0.001, 0.005, 0.01, 0.02, 0.05]  # 0.1%, 0.5%, 1%, 2%, 5%
    op_rows = []
    for fpr_t in fpr_targets:
        rec, prec, thr, k = operating_point_under_fpr(y_test, s_test, fpr_t)
        op_rows.append({
            "target_fpr": fpr_t,
            "threshold": thr,
            "k_flagged": k,
            "recall": rec,
            "precision": prec,
        })

    op_df = pd.DataFrame(op_rows)
    print("\nRecall at fixed FPR budgets (rank-based sweep):")
    print(op_df.to_string(index=False))

    # Top-K capture
    top_fracs = [0.001, 0.002, 0.005, 0.01, 0.02]  # top 0.1%, 0.2%, 0.5%, 1%, 2%
    topk = [topk_capture(y_test, s_test, f) for f in top_fracs]
    topk_df = pd.DataFrame(topk)
    print("\nTop-K capture (by score):")
    print(topk_df.to_string(index=False))

    # Save report
    out_dir = repo_root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "test_rows": int(len(y_test)),
        "test_fraud_count": int(y_test.sum()),
        "test_fraud_rate": float(y_test.mean()),
        "recall_at_fpr": op_rows,
        "topk_capture": topk,
    }
    out_path = out_dir / "operating_points_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved: {out_path}")

    # Transaction types in top 1%
    test_df = test_df.copy()
    test_df["score"] = s_test
    top_1pct = test_df.nlargest(max(1, int(0.01 * len(test_df))), "score")
    print("\nTop 1% score transaction types:")
    print(top_1pct["type"].value_counts().to_string())


if __name__ == "__main__":
    main()
