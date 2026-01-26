from __future__ import annotations
from pathlib import Path
import pandas as pd

# Windows in "steps" (PaySim step ~ hour)
W_24H = 24
W_7D = 24 * 7
W_30D = 24 * 30


def main():
    repo_root = Path(__file__).resolve().parents[2]
    in_path = repo_root / "data" / "raw" / "paysim.parquet"
    out_path = repo_root / "data" / "processed" / "paysim_with_temporal_graph_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path).copy()
    df = df.sort_values("step").reset_index(drop=True)

    # We will fill these columns
    feat_cols = [
        "orig_out_cnt_24h", "orig_out_sum_24h", "orig_out_cnt_7d", "orig_out_sum_7d",
        "orig_distinct_dest_7d",
        "dest_in_cnt_24h", "dest_in_sum_24h", "dest_in_cnt_7d", "dest_in_sum_7d",
        "dest_distinct_orig_7d",
        "pair_cnt_7d",
        "is_new_counterparty_30d",
    ]
    # Int features
    int_feats = [
        "orig_out_cnt_24h", "orig_out_cnt_7d",
        "orig_distinct_dest_7d",
        "dest_in_cnt_24h", "dest_in_cnt_7d",
        "dest_distinct_orig_7d",
        "pair_cnt_7d",
        "is_new_counterparty_30d",
    ]
    # Float features
    float_feats = [
        "orig_out_sum_24h", "orig_out_sum_7d",
        "dest_in_sum_24h", "dest_in_sum_7d",
    ]

    for c in int_feats:
        df[c] = 0
        df[c] = df[c].astype("int32")

    for c in float_feats:
        df[c] = 0.0
        df[c] = df[c].astype("float32")


    # Keep per-entity event history as lists of (step, amount, counterparty)
    orig_hist = {}  # orig -> list of (step, amount, dest)
    dest_hist = {}  # dest -> list of (step, amount, orig)
    pair_hist = {}  # (orig, dest) -> list of steps

    def window_stats(events, current_step, window):
        """events is list of tuples (step, amount, counterparty). Return count and sum in window."""
        if not events:
            return 0, 0.0, []
        cutoff = current_step - window
        # keep only events with step >= cutoff and < current_step
        recent = [e for e in events if cutoff <= e[0] < current_step]
        cnt = len(recent)
        s = float(sum(e[1] for e in recent)) if recent else 0.0
        return cnt, s, recent

    for i, row in df.iterrows():
        t = int(row["step"])
        orig = row["nameOrig"]
        dest = row["nameDest"]
        amt = float(row["amount"])

        oh = orig_hist.get(orig, [])
        dh = dest_hist.get(dest, [])
        ph = pair_hist.get((orig, dest), [])

        # orig outgoing
        c24, s24, recent24 = window_stats(oh, t, W_24H)
        c7, s7, recent7 = window_stats(oh, t, W_7D)
        df.at[i, "orig_out_cnt_24h"] = c24
        df.at[i, "orig_out_sum_24h"] = s24
        df.at[i, "orig_out_cnt_7d"] = c7
        df.at[i, "orig_out_sum_7d"] = s7
        df.at[i, "orig_distinct_dest_7d"] = len(set(e[2] for e in recent7)) if recent7 else 0

        # dest incoming
        dc24, ds24, drecent24 = window_stats(dh, t, W_24H)
        dc7, ds7, drecent7 = window_stats(dh, t, W_7D)
        df.at[i, "dest_in_cnt_24h"] = dc24
        df.at[i, "dest_in_sum_24h"] = ds24
        df.at[i, "dest_in_cnt_7d"] = dc7
        df.at[i, "dest_in_sum_7d"] = ds7
        df.at[i, "dest_distinct_orig_7d"] = len(set(e[2] for e in drecent7)) if drecent7 else 0

        # pair history in 7d
        cutoff7 = t - W_7D
        pair_recent7 = [s for s in ph if cutoff7 <= s < t]
        df.at[i, "pair_cnt_7d"] = len(pair_recent7)

        # new counterparty in last 30d?
        cutoff30 = t - W_30D
        seen_recent30 = any((cutoff30 <= e[0] < t and e[2] == dest) for e in oh)
        df.at[i, "is_new_counterparty_30d"] = 0 if seen_recent30 else 1

        # Update histories AFTER computing features (prevents leakage)
        orig_hist.setdefault(orig, []).append((t, amt, dest))
        dest_hist.setdefault(dest, []).append((t, amt, orig))
        pair_hist.setdefault((orig, dest), []).append(t)

        if i > 0 and i % 50000 == 0:
            print(f"processed {i:,}/{len(df):,}")

    df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("Added feature columns:", feat_cols)


if __name__ == "__main__":
    main()
