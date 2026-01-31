from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch


# =========================
# CONFIG
# =========================
# Use behavioral engineered features (from your processed parquet) as the TGN "message".
USE_BEHAVIORAL_FEATURES = True

# Keep this False to avoid synthetic proxy-label features:
# (Do NOT add orig_error/dest_error/etc here.)
USE_BALANCES_IN_MSG = False

# If True: log1p transforms most heavy-tailed numeric features for stability
USE_LOG1P = True
# =========================


def _log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x, a_min=0.0, a_max=None))


def main():
    repo_root = Path(__file__).resolve().parents[2]
    in_path = repo_root / "data" / "processed" / "paysim_with_temporal_graph_features.parquet"
    out_dir = repo_root / "data" / "gnn"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing: {in_path}")

    df = pd.read_parquet(in_path).sort_values("step").reset_index(drop=True)

    # -------------------------
    # Node id mapping
    # -------------------------
    all_names = pd.concat([df["nameOrig"], df["nameDest"]], ignore_index=True)
    codes, uniques = pd.factorize(all_names, sort=False)

    n = len(df)
    src = codes[:n].astype(np.int64)
    dst = codes[n:].astype(np.int64)
    num_nodes = int(uniques.size)

    # -------------------------
    # Time + label
    # -------------------------
    t = df["step"].astype(np.int64).to_numpy()
    y = df["isFraud"].astype(np.int64).to_numpy()

    # -------------------------
    # Type encoding (one-hot)
    # -------------------------
    types = df["type"].astype(str)
    type_vals = sorted(types.unique().tolist())
    type_to_id = {k: i for i, k in enumerate(type_vals)}
    type_id = types.map(type_to_id).astype(np.int64).to_numpy()
    type_oh = np.eye(len(type_vals), dtype=np.float32)[type_id]

    # -------------------------
    # Base edge features
    # -------------------------
    amount = df["amount"].astype(np.float32).to_numpy()
    f_amount = _log1p(amount) if USE_LOG1P else amount
    f_amount = f_amount.reshape(-1, 1).astype(np.float32)

    msg_parts = [f_amount, type_oh]
    msg_names = ["log1p_amount" if USE_LOG1P else "amount"] + [f"type_{k}" for k in type_vals]

    # -------------------------
    # Optional: balances (not proxy-label errors)
    # -------------------------
    if USE_BALANCES_IN_MSG:
        for col in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
            v = df[col].astype(np.float32).to_numpy()
            v = _log1p(v) if USE_LOG1P else v
            msg_parts.append(v.reshape(-1, 1))
            msg_names.append(f"log1p_{col}" if USE_LOG1P else col)

    # -------------------------
    # Hybrid behavioral features (already computed causally in your feature builder)
    # -------------------------
    if USE_BEHAVIORAL_FEATURES:
        # These are the ones that usually matter most and are stable to include:
        beh_cols = [
            # velocity
            "orig_out_cnt_1h", "orig_out_sum_1h",
            "orig_out_cnt_24h", "orig_out_sum_24h",
            "orig_out_cnt_7d", "orig_out_sum_7d",
            "dest_in_cnt_1h", "dest_in_sum_1h",
            "dest_in_cnt_24h", "dest_in_sum_24h",
            "dest_in_cnt_7d", "dest_in_sum_7d",

            # diversity / counterparties
            "orig_distinct_dest_7d",
            "dest_distinct_orig_7d",

            # pair + novelty
            "pair_cnt_7d",
            "is_new_counterparty_30d",

            # time since last
            "orig_time_since_last",
            "dest_time_since_last",
            "pair_time_since_last",

            # burstiness
            "orig_burst_cnt_1h_vs_7d",
            "dest_burst_cnt_1h_vs_7d",
        ]

        missing = [c for c in beh_cols if c not in df.columns]
        if missing:
            raise ValueError(
                "Missing behavioral columns in processed parquet. "
                "Did you run src/features/build_temporal_graph_features.py?\n"
                f"Missing: {missing}"
            )

        for col in beh_cols:
            v = df[col].astype(np.float32).to_numpy()

            # log1p for count/sum/time features; for ratios (burstiness) we also log1p after clipping
            if USE_LOG1P:
                if "burst" in col:
                    # burstiness can be >1 and heavy-tailed; clip and log1p
                    v = _log1p(np.clip(v, 0.0, 1e6))
                    msg_names.append(f"log1p_{col}")
                elif ("cnt" in col) or ("sum" in col) or ("time_since" in col) or ("distinct" in col) or ("pair_cnt" in col):
                    v = _log1p(v)
                    msg_names.append(f"log1p_{col}")
                else:
                    # binary/other small-range features
                    msg_names.append(col)
            else:
                msg_names.append(col)

            msg_parts.append(v.reshape(-1, 1))

    msg = np.concatenate(msg_parts, axis=1).astype(np.float32)

    payload = {
        "src": torch.from_numpy(src),
        "dst": torch.from_numpy(dst),
        "t": torch.from_numpy(t),
        "msg": torch.from_numpy(msg),
        "y": torch.from_numpy(y),
        "type_id": torch.from_numpy(type_id),
        "meta": {
            "num_nodes": num_nodes,
            "num_events": int(len(df)),
            "msg_dim": int(msg.shape[1]),
            "msg_names": msg_names,
            "type_values": type_vals,
            "use_behavioral_features": bool(USE_BEHAVIORAL_FEATURES),
            "use_balances_in_msg": bool(USE_BALANCES_IN_MSG),
            "use_log1p": bool(USE_LOG1P),
        },
    }

    out_path = out_dir / "paysim_tgn_events.pt"
    torch.save(payload, out_path)

    map_path = out_dir / "node_mapping.csv.gz"
    pd.DataFrame({"node_id": np.arange(num_nodes, dtype=np.int64), "name": uniques.astype(str)}).to_csv(
        map_path, index=False
    )

    meta_path = out_dir / "paysim_tgn_meta.json"
    meta_path.write_text(json.dumps(payload["meta"], indent=2))

    print("Saved:", out_path)
    print("Saved:", map_path)
    print("Saved:", meta_path)
    print("num_nodes:", num_nodes)
    print("events:", len(df))
    print("fraud_rate:", float(y.mean()))
    print("msg_dim:", msg.shape[1])
    print("first msg names:", msg_names[:12], "...")


if __name__ == "__main__":
    main()
