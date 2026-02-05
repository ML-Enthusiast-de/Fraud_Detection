from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch


# =========================
# CONFIG
# =========================
TRAIN_FRAC_FOR_SCALER = 0.70  # fit standardization only on early train window (no leakage)
EPS = 1e-8

# Preferred input (has temporal graph behavior features already)
PROCESSED_WITH_GRAPH = "data/processed/paysim_with_temporal_graph_features.parquet"

# Fallback inputs (if you ever need them)
FALLBACK_CANDIDATES = [
    "data/processed/paysim_sample.parquet",
    "data/raw/paysim_sample.parquet",
    "data/raw/paysim_sample.csv",
    "data/raw/paysim.csv",
]

# Output for run_21
OUT_PATH = "data/gnn/paysim_tgn_events.pt"
# =========================


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """sign(x) * log1p(abs(x))"""
    return np.sign(x) * np.log1p(np.abs(x))


def safe_log1p(x: np.ndarray) -> np.ndarray:
    """log1p(max(x,0)) for non-negative features"""
    return np.log1p(np.maximum(x, 0.0))


def _first_existing(repo_root: Path, rel_paths: list[str]) -> Path | None:
    for rp in rel_paths:
        p = repo_root / rp
        if p.exists():
            return p
    return None


def main():
    repo_root = Path(__file__).resolve().parents[2]

    # ---------- Load dataset ----------
    preferred = repo_root / PROCESSED_WITH_GRAPH
    if preferred.exists():
        df = pd.read_parquet(preferred)
        source_path = preferred
    else:
        fallback = _first_existing(repo_root, FALLBACK_CANDIDATES)
        if fallback is None:
            raise FileNotFoundError(
                "Could not find a PaySim file.\n"
                f"Expected either:\n  - {preferred}\n"
                f"or one of:\n  - " + "\n  - ".join(str(repo_root / p) for p in FALLBACK_CANDIDATES)
            )
        if fallback.suffix.lower() == ".csv":
            df = pd.read_csv(fallback)
        else:
            df = pd.read_parquet(fallback)
        source_path = fallback
        print(f"[WARN] Using fallback input without temporal-graph features: {source_path}")
        print("[WARN] For best results, build temporal graph features first and save to:")
        print(f"       {preferred}")

    required = [
        "step", "type", "amount",
        "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest",
        "isFraud",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure deterministic time order
    df = df.sort_values("step", kind="mergesort").reset_index(drop=True)

    # ---------- Core columns ----------
    step = df["step"].astype(np.int64).to_numpy()
    y = df["isFraud"].astype(np.int64).to_numpy()

    # type_id and type one-hot in msg
    type_values = sorted(df["type"].astype(str).unique().tolist())
    type_to_id = {k: i for i, k in enumerate(type_values)}
    type_id = df["type"].astype(str).map(type_to_id).astype(np.int64).to_numpy()
    type_oh = np.eye(len(type_values), dtype=np.float32)[type_id]  # [N, n_types]

    # node ids (accounts)
    all_names = pd.concat([df["nameOrig"].astype(str), df["nameDest"].astype(str)], axis=0)
    nodes, _ = pd.factorize(all_names, sort=False)
    n = len(df)
    src = nodes[:n].astype(np.int64)
    dst = nodes[n:].astype(np.int64)
    num_nodes = int(nodes.max()) + 1

    # ---------- Mechanics features ----------
    amount = df["amount"].astype(np.float32).to_numpy()

    old_o = df["oldbalanceOrg"].astype(np.float32).to_numpy()
    new_o = df["newbalanceOrig"].astype(np.float32).to_numpy()
    old_d = df["oldbalanceDest"].astype(np.float32).to_numpy()
    new_d = df["newbalanceDest"].astype(np.float32).to_numpy()

    # balance deltas
    orig_delta = (new_o - old_o).astype(np.float32)
    dest_delta = (new_d - old_d).astype(np.float32)

    # consistency errors (PaySim is VERY sensitive to these)
    # For a normal debit: new_o = old_o - amount  => orig_delta + amount ~ 0
    orig_error = (orig_delta + amount).astype(np.float32)
    abs_orig_error = np.abs(orig_error).astype(np.float32)

    # For a normal credit: new_d = old_d + amount => dest_delta - amount ~ 0
    dest_error = (dest_delta - amount).astype(np.float32)
    abs_dest_error = np.abs(dest_error).astype(np.float32)

    # ---------- Behavioral / temporal graph features ----------
    # We include what’s available; missing ones are filled with zeros.
    # (This keeps the script robust even if your feature builder changes.)
    behavioral_candidates = [
        # 24h / 7d “rolling” stats
        "orig_out_cnt_24h", "orig_out_sum_24h", "orig_out_cnt_7d", "orig_out_sum_7d", "orig_distinct_dest_7d",
        "dest_in_cnt_24h",  "dest_in_sum_24h",  "dest_in_cnt_7d",  "dest_in_sum_7d",  "dest_distinct_orig_7d",
        "pair_cnt_7d", "is_new_counterparty_30d",
        # optional extras you may have added later
        "orig_out_cnt_1h", "orig_out_sum_1h", "dest_in_cnt_1h", "dest_in_sum_1h",
        "orig_time_since_last", "dest_time_since_last", "pair_time_since_last",
        "orig_burst_cnt_1h_vs_7d", "dest_burst_cnt_1h_vs_7d",
    ]

    behavioral = {}
    missing_beh = []
    for c in behavioral_candidates:
        if c in df.columns:
            behavioral[c] = df[c].to_numpy()
        else:
            missing_beh.append(c)
            behavioral[c] = np.zeros(len(df), dtype=np.float32)

    if missing_beh:
        print(f"[INFO] Behavioral columns missing (filled with zeros): {missing_beh}")

    # ---------- Build msg matrix ----------
    # We apply sensible transforms before standardization:
    # - money-like: log1p
    # - deltas/errors: signed_log1p
    # - counts: log1p
    # - times: log1p
    # - ratios: clipped then log1p
    feature_blocks = []
    feature_names = []

    def add_block(arr: np.ndarray, names: list[str]):
        assert arr.shape[0] == len(df)
        feature_blocks.append(arr.astype(np.float32))
        feature_names.extend(names)

    # money-like
    add_block(
        np.stack(
            [
                safe_log1p(amount),
                safe_log1p(old_o), safe_log1p(new_o),
                safe_log1p(old_d), safe_log1p(new_d),
            ],
            axis=1,
        ),
        ["log_amount", "log_oldbalanceOrg", "log_newbalanceOrig", "log_oldbalanceDest", "log_newbalanceDest"],
    )

    # deltas/errors (signed)
    add_block(
        np.stack(
            [
                signed_log1p(orig_delta),
                signed_log1p(dest_delta),
                signed_log1p(orig_error),
                signed_log1p(dest_error),
                safe_log1p(abs_orig_error),
                safe_log1p(abs_dest_error),
            ],
            axis=1,
        ),
        ["slog_orig_delta", "slog_dest_delta", "slog_orig_error", "slog_dest_error", "log_abs_orig_error", "log_abs_dest_error"],
    )

    # behavioral: we treat sums as money-like, counts as count-like, booleans as-is
    # Identify by name:
    beh_cols = behavioral_candidates
    beh_mat = []
    beh_names = []

    for c in beh_cols:
        v = behavioral[c].astype(np.float32)

        if c.startswith("orig_out_sum") or c.startswith("dest_in_sum"):
            v = safe_log1p(v)
            beh_names.append(f"log_{c}")
        elif c.endswith("_cnt_24h") or c.endswith("_cnt_7d") or c.endswith("_cnt_1h") or c == "pair_cnt_7d":
            v = safe_log1p(v)
            beh_names.append(f"log_{c}")
        elif "time_since_last" in c:
            v = safe_log1p(v)
            beh_names.append(f"log_{c}")
        elif "burst" in c:
            # ratios can spike; clip then log1p
            v = np.clip(v, 0.0, 50.0)
            v = safe_log1p(v)
            beh_names.append(f"log_{c}_clip50")
        elif c.startswith("orig_distinct") or c.startswith("dest_distinct"):
            v = safe_log1p(v)
            beh_names.append(f"log_{c}")
        elif c == "is_new_counterparty_30d":
            # keep 0/1
            v = v.astype(np.float32)
            beh_names.append(c)
        else:
            # safe default
            v = v.astype(np.float32)
            beh_names.append(c)

        beh_mat.append(v.reshape(-1, 1))

    add_block(np.concatenate(beh_mat, axis=1), beh_names)

    # type one-hot (do NOT standardize)
    add_block(type_oh, [f"type_oh_{t}" for t in type_values])

    X = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    msg_dim = int(X.shape[1])

    # ---------- Standardize continuous features (fit on early train window only) ----------
    # We exclude one-hot columns from standardization.
    # Find the type one-hot indices:
    n_types = len(type_values)
    type_start = msg_dim - n_types
    type_idx = np.arange(type_start, msg_dim)

    cont_idx = np.array([i for i in range(msg_dim) if i not in set(type_idx)], dtype=np.int64)

    n_train = int(TRAIN_FRAC_FOR_SCALER * len(df))
    train_slice = slice(0, max(1, n_train))

    mu = X[train_slice][:, cont_idx].mean(axis=0)
    sd = X[train_slice][:, cont_idx].std(axis=0) + EPS

    X_scaled = X.copy()
    X_scaled[:, cont_idx] = (X_scaled[:, cont_idx] - mu) / sd

    # ---------- Save torch payload ----------
    out_path = repo_root / OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "src": torch.from_numpy(src).long(),
        "dst": torch.from_numpy(dst).long(),
        "t": torch.from_numpy(step).long(),
        "msg": torch.from_numpy(X_scaled).float(),
        "y": torch.from_numpy(y).long(),
        "type_id": torch.from_numpy(type_id).long(),
        "meta": {
            "source_path": str(source_path),
            "num_nodes": int(num_nodes),
            "num_events": int(len(df)),
            "msg_dim": int(msg_dim),
            "type_values": type_values,
            "feature_names": feature_names,
            "scaler": {
                "train_frac_for_scaler": TRAIN_FRAC_FOR_SCALER,
                "continuous_feature_indices": cont_idx.tolist(),
                "continuous_mu": mu.astype(np.float32).tolist(),
                "continuous_sd": sd.astype(np.float32).tolist(),
                "type_onehot_start_index": int(type_start),
            },
        },
    }

    torch.save(payload, out_path)

    print("\n=== Saved TGN events ===")
    print("out:", out_path)
    print("events:", len(df), "| nodes:", num_nodes)
    print("msg_dim:", msg_dim, "| types:", type_values)
    print("fraud:", int(y.sum()), f"({float(y.mean()):.6f})")
    print("scaler_fit_events:", max(1, n_train))
    print("========================\n")

    # Small sanity preview
    preview = {
        "first_feature_names": feature_names[: min(10, len(feature_names))],
        "last_feature_names": feature_names[max(0, len(feature_names) - 10):],
    }
    (repo_root / "artifacts").mkdir(exist_ok=True)
    (repo_root / "artifacts" / "tgn_msg_schema_preview.json").write_text(json.dumps(preview, indent=2))
    print("Wrote schema preview: artifacts/tgn_msg_schema_preview.json")


if __name__ == "__main__":
    main()
