from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch


# =========================
# CONFIG
# =========================
# Keep messages small and "production-ish":
# - log(amount)
# - one-hot(type)
USE_BALANCES_IN_MSG = False  # set True if you want, but start False
# =========================


def main():
    repo_root = Path(__file__).resolve().parents[2]
    in_path = repo_root / "data" / "processed" / "paysim_with_temporal_graph_features.parquet"
    out_dir = repo_root / "data" / "gnn"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing: {in_path}")

    df = pd.read_parquet(in_path).sort_values("step").reset_index(drop=True)

    # Map accounts to node ids
    all_names = pd.concat([df["nameOrig"], df["nameDest"]], ignore_index=True)
    codes, uniques = pd.factorize(all_names, sort=False)
    n = len(df)
    src = codes[:n].astype(np.int64)
    dst = codes[n:].astype(np.int64)
    num_nodes = int(uniques.size)

    # Time and label
    t = df["step"].astype(np.int64).to_numpy()
    y = df["isFraud"].astype(np.int64).to_numpy()

    # Build message features
    types = df["type"].astype(str)
    type_vals = sorted(types.unique().tolist())
    type_to_id = {k: i for i, k in enumerate(type_vals)}
    type_id = types.map(type_to_id).astype(np.int64).to_numpy()

    # one-hot(type)
    type_oh = np.eye(len(type_vals), dtype=np.float32)[type_id]

    # log1p(amount)
    log_amount = np.log1p(df["amount"].astype(np.float32).to_numpy()).reshape(-1, 1)

    msg_parts = [log_amount, type_oh]

    if USE_BALANCES_IN_MSG:
        # log1p(balances) to keep scales sane
        for col in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
            msg_parts.append(np.log1p(df[col].astype(np.float32).to_numpy()).reshape(-1, 1))

    msg = np.concatenate(msg_parts, axis=1).astype(np.float32)

    # Save as torch tensors
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
            "type_values": type_vals,
            "use_balances_in_msg": bool(USE_BALANCES_IN_MSG),
        },
    }

    out_path = out_dir / "paysim_tgn_events.pt"
    torch.save(payload, out_path)

    # Save mapping + meta
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
    print("types:", type_vals)


if __name__ == "__main__":
    main()
