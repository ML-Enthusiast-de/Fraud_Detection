from __future__ import annotations

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="X does not have valid feature names*")


ALLOWED_TYPES = {"TRANSFER", "CASH_OUT"}  # stage-0 gate


def load_scores(repo_root: Path):
    import joblib

    data_path = repo_root / "data" / "processed" / "paysim_with_temporal_graph_features.parquet"
    model_path = repo_root / "artifacts" / "temporal_graph_lgbm.joblib"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run feature builder first.")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}. Train temporal graph model first.")

    df = pd.read_parquet(data_path).sort_values("step").reset_index(drop=True)

    # same split logic as training: 70/15/15
    n = len(df)
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
    choose max recall such that FPR <= target_fpr.
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


def run_eval(tag: str, test_df: pd.DataFrame, y_test: np.ndarray, score: np.ndarray):
    print(f"\n=== {tag} ===")
    print("Test rows:", len(y_test))
    print("Test fraud count:", int(y_test.sum()))
    print("Test fraud rate:", float(y_test.mean()))

    fpr_targets = [0.001, 0.005, 0.01, 0.02, 0.05]
    op_rows = []
    for fpr_t in fpr_targets:
        rec, prec, thr, k = operating_point_under_fpr(y_test, score, fpr_t)
        op_rows.append({
            "target_fpr": fpr_t,
            "threshold": thr,
            "k_flagged": k,
            "recall": rec,
            "precision": prec,
        })
    op_df = pd.DataFrame(op_rows)
    print("\nRecall at fixed FPR budgets:")
    print(op_df.to_string(index=False))

    top_fracs = [0.001, 0.002, 0.005, 0.01, 0.02]
    topk = [topk_capture(y_test, score, f) for f in top_fracs]
    topk_df = pd.DataFrame(topk)
    print("\nTop-K capture (by score):")
    print(topk_df.to_string(index=False))

    tmp = test_df.copy()
    tmp["score"] = score
    top_1pct = tmp.nlargest(max(1, int(0.01 * len(tmp))), "score")
    print("\nTop 1% score transaction types:")
    print(top_1pct["type"].value_counts().to_string())

    return {"recall_at_fpr": op_rows, "topk_capture": topk}


def main():
    repo_root = Path(__file__).resolve().parents[2]
    test_df, y_test, s_test = load_scores(repo_root)

    # 1) Raw (no gate)
    report_raw = run_eval("RAW (no gating)", test_df, y_test, s_test)

    # 2) Gated: only score TRANSFER/CASH_OUT, else score=0
    mask = test_df["type"].isin(ALLOWED_TYPES).values
    s_gated = s_test.copy()
    s_gated[~mask] = 0.0

    report_gated = run_eval(f"GATED (types in {sorted(ALLOWED_TYPES)})", test_df, y_test, s_gated)

    out = {
        "allowed_types": sorted(ALLOWED_TYPES),
        "raw": report_raw,
        "gated": report_gated,
    }
    out_path = repo_root / "artifacts" / "operating_points_report_gated.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
