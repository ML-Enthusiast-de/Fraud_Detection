from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

# PaySim step ~ hour
W_1H = 1
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

    # --- Feature columns to add ---
    int_feats = [
        # 1h / 24h / 7d velocity counts
        "orig_out_cnt_1h", "orig_out_cnt_24h", "orig_out_cnt_7d",
        "dest_in_cnt_1h", "dest_in_cnt_24h", "dest_in_cnt_7d",
        # distinct counterparties
        "orig_distinct_dest_7d",
        "dest_distinct_orig_7d",
        # pair history
        "pair_cnt_7d",
        "is_new_counterparty_30d",
        # time-since (in steps)
        "orig_time_since_last",
        "dest_time_since_last",
        "pair_time_since_last",
    ]

    float_feats = [
        # 1h / 24h / 7d velocity sums
        "orig_out_sum_1h", "orig_out_sum_24h", "orig_out_sum_7d",
        "dest_in_sum_1h", "dest_in_sum_24h", "dest_in_sum_7d",
        # burstiness ratios
        "orig_burst_cnt_1h_vs_7d",
        "dest_burst_cnt_1h_vs_7d",
    ]

    # Initialize with correct dtypes (avoids pandas FutureWarning)
    for c in int_feats:
        df[c] = np.int32(0)
    for c in float_feats:
        df[c] = np.float32(0.0)

    # Per-entity histories: list of (step, amount, counterparty)
    orig_hist: dict[str, list[tuple[int, float, str]]] = {}
    dest_hist: dict[str, list[tuple[int, float, str]]] = {}
    pair_steps: dict[tuple[str, str], list[int]] = {}

    # Last seen step (for time-since-last)
    orig_last: dict[str, int] = {}
    dest_last: dict[str, int] = {}
    pair_last: dict[tuple[str, str], int] = {}

    def window_stats(events, current_step: int, window: int):
        """events: list[(step, amount, counterparty)] -> count, sum, recent_events"""
        if not events:
            return 0, 0.0, []
        cutoff = current_step - window
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
        ph = pair_steps.get((orig, dest), [])

        # -------- time since last (no leakage: use last seen < t) --------
        df.at[i, "orig_time_since_last"] = np.int32(t - orig_last[orig]) if orig in orig_last else np.int32(10_000)
        df.at[i, "dest_time_since_last"] = np.int32(t - dest_last[dest]) if dest in dest_last else np.int32(10_000)
        df.at[i, "pair_time_since_last"] = np.int32(t - pair_last[(orig, dest)]) if (orig, dest) in pair_last else np.int32(10_000)

        # -------- orig outgoing windows --------
        c1, s1, _ = window_stats(oh, t, W_1H)
        c24, s24, _ = window_stats(oh, t, W_24H)
        c7, s7, recent7 = window_stats(oh, t, W_7D)

        df.at[i, "orig_out_cnt_1h"] = np.int32(c1)
        df.at[i, "orig_out_sum_1h"] = np.float32(s1)
        df.at[i, "orig_out_cnt_24h"] = np.int32(c24)
        df.at[i, "orig_out_sum_24h"] = np.float32(s24)
        df.at[i, "orig_out_cnt_7d"] = np.int32(c7)
        df.at[i, "orig_out_sum_7d"] = np.float32(s7)
        df.at[i, "orig_distinct_dest_7d"] = np.int32(len(set(e[2] for e in recent7)) if recent7 else 0)

        # burstiness: compare 1h activity vs average hourly activity over 7d
        avg_hourly_7d = c7 / float(W_7D)  # per-hour average
        df.at[i, "orig_burst_cnt_1h_vs_7d"] = np.float32((c1 + 1.0) / (avg_hourly_7d + 1.0))

        # -------- dest incoming windows --------
        dc1, ds1, _ = window_stats(dh, t, W_1H)
        dc24, ds24, _ = window_stats(dh, t, W_24H)
        dc7, ds7, drecent7 = window_stats(dh, t, W_7D)

        df.at[i, "dest_in_cnt_1h"] = np.int32(dc1)
        df.at[i, "dest_in_sum_1h"] = np.float32(ds1)
        df.at[i, "dest_in_cnt_24h"] = np.int32(dc24)
        df.at[i, "dest_in_sum_24h"] = np.float32(ds24)
        df.at[i, "dest_in_cnt_7d"] = np.int32(dc7)
        df.at[i, "dest_in_sum_7d"] = np.float32(ds7)
        df.at[i, "dest_distinct_orig_7d"] = np.int32(len(set(e[2] for e in drecent7)) if drecent7 else 0)

        avg_hourly_dest_7d = dc7 / float(W_7D)
        df.at[i, "dest_burst_cnt_1h_vs_7d"] = np.float32((dc1 + 1.0) / (avg_hourly_dest_7d + 1.0))

        # -------- pair history --------
        cutoff7 = t - W_7D
        pair_recent7 = [s for s in ph if cutoff7 <= s < t]
        df.at[i, "pair_cnt_7d"] = np.int32(len(pair_recent7))

        # -------- new counterparty? (orig->dest unseen in last 30d) --------
        cutoff30 = t - W_30D
        seen_recent30 = any((cutoff30 <= e[0] < t and e[2] == dest) for e in oh)
        df.at[i, "is_new_counterparty_30d"] = np.int32(0 if seen_recent30 else 1)

        # -------- update histories AFTER feature computation (prevents leakage) --------
        orig_hist.setdefault(orig, []).append((t, amt, dest))
        dest_hist.setdefault(dest, []).append((t, amt, orig))
        pair_steps.setdefault((orig, dest), []).append(t)

        orig_last[orig] = t
        dest_last[dest] = t
        pair_last[(orig, dest)] = t

        if i > 0 and i % 50_000 == 0:
            print(f"processed {i:,}/{len(df):,}")

    df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("Added int features:", int_feats)
    print("Added float features:", float_feats)


if __name__ == "__main__":
    main()
