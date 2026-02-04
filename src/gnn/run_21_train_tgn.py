from __future__ import annotations

from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import (
    TGNMemory,
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

# =========================
# CONFIG (tune here)
# =========================
MODEL_VARIANT = "behavioral"

# ---- SPLIT STRATEGY (KEY CHANGE) ----
# Keep TRAIN as the first TRAIN_FRAC of time.
# Then, within the remaining tail, choose the VAL/TEST boundary so TEST contains
# the last TEST_FRAUD_TARGET frauds. This moves fraud into VAL (less noisy selection).
TRAIN_FRAC = 0.70

TEST_FRAUD_TARGET = 100     # how many frauds to keep in TEST (from the tail end)
MIN_VAL_FRAUD = 80          # ensure VAL gets at least this many frauds (auto-adjust test target if needed)
MIN_TEST_EVENTS = 10_000    # safety: avoid absurdly tiny test window (auto-adjust if needed)

BATCH_SIZE = 2000
EPOCHS = 25
LR = 1e-3

MEMORY_DIM = 128
TIME_DIM = 64
HEADS = 2
EMB_DIM = 128  # must be divisible by HEADS

NEIGHBOR_SIZE = 50

# Regularization / stability
CLIP_GRAD_NORM = 1.0
WEIGHT_DECAY = 1e-4

# Loss
USE_FOCAL = True
FOCAL_GAMMA = 2.0

# Checkpoint selection: weighted recall at fixed FPR budgets on VAL
SELECT_FPRS = (0.005, 0.01, 0.02, 0.05)      # 0.5%, 1%, 2%, 5%
SELECT_WEIGHTS = (0.45, 0.25, 0.20, 0.10)    # sum to 1

EARLY_STOP_PATIENCE = 6
MIN_EPOCHS_BEFORE_STOP = 3

# Optional ops-style gate for reporting (does NOT affect training)
ALLOWED_TYPES_GATE = {"TRANSFER", "CASH_OUT"}

SEED = 42
# =========================


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal loss on top of BCEWithLogits, with pos_weight for class imbalance.
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    p = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, p, 1.0 - p)
    mod = (1.0 - pt).pow(gamma)
    return (mod * bce).mean()


class TemporalGraphEmbedding(nn.Module):
    """Neighborhood GNN on top of node memory, with edge attributes."""
    def __init__(self, in_dim: int, emb_dim: int, edge_dim: int, heads: int):
        super().__init__()
        assert emb_dim % heads == 0
        out_per_head = emb_dim // heads
        self.conv = TransformerConv(
            in_channels=in_dim,
            out_channels=out_per_head,
            heads=heads,
            concat=True,
            dropout=0.1,
            edge_dim=edge_dim,
        )

    def forward(self, x, edge_index, edge_attr):
        return self.conv(x, edge_index, edge_attr)


class EdgeClassifier(nn.Module):
    """Binary classifier on (src_emb, dst_emb, current_msg)."""
    def __init__(self, emb_dim: int, msg_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * emb_dim + msg_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, z_src, z_dst, msg):
        x = torch.cat([z_src, z_dst, msg], dim=-1)
        return self.net(x).view(-1)


def split_train_tail_by_fraud(
    y: np.ndarray,
    n_events: int,
    train_frac: float,
    test_fraud_target: int,
    min_val_fraud: int,
    min_test_events: int,
):
    """
    Time-ordered split:

    TRAIN = [0, i1)
    VAL   = [i1, i2)
    TEST  = [i2, end)

    Where i1 is fixed by train_frac, and i2 is chosen so that TEST contains the last
    `test_fraud_target` frauds from the tail [i1, end).

    This specifically fixes PaySim's "fraud spike at the very end" problem.
    """
    assert 0.0 < train_frac < 1.0
    i1 = int(train_frac * n_events)

    y_tail = y[i1:]
    tail_fraud_idx = np.flatnonzero(y_tail == 1)
    tail_fraud = int(len(tail_fraud_idx))

    if tail_fraud == 0:
        # No fraud in tail: fallback to simple 70/20/10 style split
        i2 = int((train_frac + 0.20) * n_events)
        return i1, i2, {
            "mode": "fallback_frac",
            "tail_fraud": tail_fraud,
            "test_fraud_target_used": 0,
        }

    # Ensure VAL isn't starved: TEST cannot take so many frauds that VAL has < min_val_fraud
    max_test_fraud = max(1, tail_fraud - min_val_fraud) if tail_fraud > min_val_fraud else max(1, tail_fraud // 2)
    test_fraud_used = int(min(test_fraud_target, max_test_fraud))

    # Pick i2 = position (in tail) where the last `test_fraud_used` frauds begin
    start_in_tail = int(tail_fraud_idx[-test_fraud_used])
    i2 = i1 + start_in_tail

    # Safety: ensure TEST has at least min_test_events; if not, move boundary earlier.
    test_events = n_events - i2
    if test_events < min_test_events:
        i2 = max(i1 + 1, n_events - min_test_events)
        # recompute how many frauds ended up in TEST after forcing minimum size
        test_fraud_after = int(y[i2:].sum())
        info = {
            "mode": "tail_fraud_with_min_test_events",
            "tail_fraud": tail_fraud,
            "test_fraud_target_used": test_fraud_used,
            "test_events_forced": True,
            "test_fraud_after_forcing": test_fraud_after,
        }
        return i1, i2, info

    info = {
        "mode": "tail_fraud",
        "tail_fraud": tail_fraud,
        "test_fraud_target_used": test_fraud_used,
        "test_events_forced": False,
    }
    return i1, i2, info


@torch.no_grad()
def rank_recall_at_fpr(y_true: np.ndarray, score: np.ndarray, target_fpr: float):
    """
    Rank-based operating point:
    choose top-k such that FPR <= target_fpr and recall is maximized within that constraint.
    """
    order = np.argsort(-score)
    y = y_true[order]
    s = score[order]

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0, float("inf"), 0, 0.0

    is_pos = (y == 1)
    is_neg = ~is_pos
    cum_tp = np.cumsum(is_pos)
    cum_fp = np.cumsum(is_neg)

    fpr = cum_fp / n_neg
    recall = cum_tp / n_pos
    precision = cum_tp / np.maximum(1, (cum_tp + cum_fp))

    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0, float("inf"), 0, 0.0

    j = valid[np.argmax(recall[valid])]
    return float(recall[j]), float(s[j]), int(j + 1), float(precision[j])


def main():
    seed_everything(SEED)

    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / "data" / "gnn" / "paysim_tgn_events.pt"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run run_20_prepare_tgn_data.py first.")

    artifacts = repo_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    payload = torch.load(data_path)
    src = payload["src"].long()
    dst = payload["dst"].long()
    t = payload["t"].long()
    msg = payload["msg"].float()
    y = payload["y"].long()
    type_id = payload["type_id"].long()
    meta = payload["meta"]

    num_nodes = int(meta["num_nodes"])
    msg_dim = int(meta["msg_dim"])
    type_values = meta.get("type_values", [])
    type_to_id = {k: i for i, k in enumerate(type_values)}
    allowed_type_ids = torch.tensor(
        [type_to_id[k] for k in ALLOWED_TYPES_GATE if k in type_to_id],
        dtype=torch.long,
    )

    data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y, type_id=type_id)

    n_events = data.num_events
    y_np_all = data.y.cpu().numpy().astype(np.int64)

    i1, i2, split_info = split_train_tail_by_fraud(
        y=y_np_all,
        n_events=n_events,
        train_frac=TRAIN_FRAC,
        test_fraud_target=TEST_FRAUD_TARGET,
        min_val_fraud=MIN_VAL_FRAUD,
        min_test_events=MIN_TEST_EVENTS,
    )

    train_data = data[:i1]
    val_data = data[i1:i2]
    test_data = data[i2:]

    def _split_stats(name: str, td: TemporalData):
        yy = td.y.cpu().numpy()
        print(f"{name}: events={len(td)} fraud={int(yy.sum())} rate={float(yy.mean()):.6f}")

    print("\n=== TIME SPLIT (TRAIN fixed, VAL/TEST tail by fraud) ===")
    print(f"TRAIN_FRAC={TRAIN_FRAC:.2f} | TEST_FRAUD_TARGET={TEST_FRAUD_TARGET} | MIN_VAL_FRAUD={MIN_VAL_FRAUD} | MIN_TEST_EVENTS={MIN_TEST_EVENTS}")
    print("split_info:", split_info)
    _split_stats("TRAIN", train_data)
    _split_stats("VAL", val_data)
    _split_stats("TEST", test_data)
    print("======================================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ---- TGN memory ----
    memory = TGNMemory(
        num_nodes=num_nodes,
        raw_msg_dim=msg_dim,
        memory_dim=MEMORY_DIM,
        time_dim=TIME_DIM,
        message_module=IdentityMessage(msg_dim, MEMORY_DIM, TIME_DIM),
        aggregator_module=LastAggregator(),
    ).to(device)

    neighbor_loader = LastNeighborLoader(num_nodes=num_nodes, size=NEIGHBOR_SIZE, device=device)

    edge_dim = msg_dim + TIME_DIM
    embedder = TemporalGraphEmbedding(MEMORY_DIM, EMB_DIM, edge_dim=edge_dim, heads=HEADS).to(device)
    classifier = EdgeClassifier(EMB_DIM, msg_dim).to(device)

    params = list(memory.parameters()) + list(embedder.parameters()) + list(classifier.parameters())
    opt = torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)

    # ---- Class imbalance on TRAIN only ----
    y_train = train_data.y.cpu().numpy()
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32, device=device)

    print("train events:", len(train_data))
    print("train pos:", n_pos, "neg:", n_neg, "pos_weight:", float(pos_weight.item()))

    train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)

    assoc = torch.empty(num_nodes, device=device, dtype=torch.long)

    def reset_state():
        memory.reset_state()
        neighbor_loader.reset_state()

    def batch_predict(batch: TemporalData, t_now: torch.Tensor):
        seed = torch.cat([batch.src, batch.dst]).unique()

        # PyG signature: __call__(n_id)
        n_id, edge_index, e_id = neighbor_loader(seed)

        mem, _ = memory(n_id)

        if edge_index.numel() == 0 or e_id.numel() == 0:
            z = mem
        else:
            e_t = batch.t_all[e_id]
            age = (t_now - e_t).clamp_min(0).to(mem.dtype)
            age_enc = memory.time_enc(age)
            e_msg = batch.msg_all[e_id]
            e_attr = torch.cat([e_msg, age_enc], dim=-1)
            z = embedder(mem, edge_index, e_attr)

        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z_src = z[assoc[batch.src]]
        z_dst = z[assoc[batch.dst]]
        logits = classifier(z_src, z_dst, batch.msg)
        return logits

    def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if USE_FOCAL:
            return focal_bce_with_logits(logits, targets, pos_weight=pos_weight, gamma=FOCAL_GAMMA)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    def train_one_epoch():
        memory.train()
        embedder.train()
        classifier.train()
        reset_state()

        train_t_all = train_data.t.to(device)
        train_msg_all = train_data.msg.to(device)

        total_loss = 0.0
        total = 0

        for batch in train_loader:
            batch = batch.to(device)
            batch.t_all = train_t_all
            batch.msg_all = train_msg_all

            opt.zero_grad(set_to_none=True)

            t_now = batch.t.max()
            logits = batch_predict(batch, t_now)
            loss = compute_loss(logits, batch.y.float())
            loss.backward()

            if CLIP_GRAD_NORM is not None and CLIP_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(params, CLIP_GRAD_NORM)

            opt.step()

            # Update AFTER scoring (no leakage)
            memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            neighbor_loader.insert(batch.src, batch.dst)
            memory.detach()

            bs = batch.num_events
            total_loss += float(loss.item()) * bs
            total += bs

        return total_loss / max(1, total)

    @torch.no_grad()
    def eval_stream(history: TemporalData, stream: TemporalData, stream_name: str):
        memory.eval()
        embedder.eval()
        classifier.eval()
        reset_state()

        all_t = torch.cat([history.t, stream.t]).to(device)
        all_msg = torch.cat([history.msg, stream.msg]).to(device)

        # warmup
        history_loader = TemporalDataLoader(history, batch_size=BATCH_SIZE)
        for batch in history_loader:
            batch = batch.to(device)
            memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            neighbor_loader.insert(batch.src, batch.dst)

        stream_loader = TemporalDataLoader(stream, batch_size=BATCH_SIZE)

        all_logits, all_y, all_type = [], [], []
        for batch in stream_loader:
            batch = batch.to(device)
            batch.t_all = all_t
            batch.msg_all = all_msg

            t_now = batch.t.max()
            logits = batch_predict(batch, t_now)

            all_logits.append(logits.detach().cpu())
            all_y.append(batch.y.detach().cpu())
            all_type.append(batch.type_id.detach().cpu())

            memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            neighbor_loader.insert(batch.src, batch.dst)

        logits = torch.cat(all_logits).numpy()
        y_true = torch.cat(all_y).numpy().astype(np.int64)
        type_ids = torch.cat(all_type).numpy().astype(np.int64)
        score = sigmoid_np(logits)

        def print_ops(tag: str, this_score: np.ndarray):
            print(f"\n=== {tag} ===")
            print("rows:", len(y_true), "fraud:", int(y_true.sum()), "rate:", float(y_true.mean()))
            recalls = {}
            for fpr_t in [0.001, 0.005, 0.01, 0.02, 0.05]:
                rec, thr, k, prec = rank_recall_at_fpr(y_true, this_score, fpr_t)
                recalls[fpr_t] = rec
                print({"target_fpr": fpr_t, "threshold": thr, "k_flagged": k, "recall": rec, "precision": prec})
            return recalls

        raw_recalls = print_ops(f"{stream_name} RAW", score)

        gated_recalls = None
        if len(allowed_type_ids) > 0:
            mask = np.isin(type_ids, allowed_type_ids.cpu().numpy())
            gated_score = score.copy()
            gated_score[~mask] = 0.0
            gated_recalls = print_ops(f"{stream_name} GATED {sorted(ALLOWED_TYPES_GATE)}", gated_score)

        return {"raw_recalls": raw_recalls, "gated_recalls": gated_recalls}

    # ---- checkpoint selection ----
    best_path = artifacts / f"tgn_{MODEL_VARIANT}.pt"
    best_score = -1.0
    best_epoch = -1
    bad_epochs = 0

    base_config = {
        "MODEL_VARIANT": MODEL_VARIANT,
        "TRAIN_FRAC": TRAIN_FRAC,
        "TEST_FRAUD_TARGET": TEST_FRAUD_TARGET,
        "MIN_VAL_FRAUD": MIN_VAL_FRAUD,
        "MIN_TEST_EVENTS": MIN_TEST_EVENTS,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "USE_FOCAL": USE_FOCAL,
        "FOCAL_GAMMA": FOCAL_GAMMA,
        "MEMORY_DIM": MEMORY_DIM,
        "TIME_DIM": TIME_DIM,
        "HEADS": HEADS,
        "EMB_DIM": EMB_DIM,
        "NEIGHBOR_SIZE": NEIGHBOR_SIZE,
        "ALLOWED_TYPES_GATE": sorted(ALLOWED_TYPES_GATE),
        "SELECT_FPRS": list(SELECT_FPRS),
        "SELECT_WEIGHTS": list(SELECT_WEIGHTS),
        "EARLY_STOP_PATIENCE": EARLY_STOP_PATIENCE,
        "MIN_EPOCHS_BEFORE_STOP": MIN_EPOCHS_BEFORE_STOP,
        "SEED": SEED,
        "CLIP_GRAD_NORM": CLIP_GRAD_NORM,
        "split_info": split_info,
    }

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch()
        print(f"epoch {epoch}/{EPOCHS} train_loss: {loss:.6f}")

        val_out = eval_stream(train_data, val_data, stream_name=f"VAL(epoch={epoch})")

        r = val_out["raw_recalls"]
        val_score = 0.0
        for fpr_t, w in zip(SELECT_FPRS, SELECT_WEIGHTS):
            val_score += float(w) * float(r.get(float(fpr_t), 0.0))

        print(f"VAL selection score (mix @ {SELECT_FPRS}): {val_score:.6f}")

        improved = val_score > best_score + 1e-12
        if improved:
            best_score = val_score
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "memory": memory.state_dict(),
                    "embedder": embedder.state_dict(),
                    "classifier": classifier.state_dict(),
                    "meta": meta,
                    "config": base_config,
                    "best_epoch": best_epoch,
                    "best_val_score": best_score,
                },
                best_path,
            )
            print("saved best by VAL score:", best_path)
        else:
            bad_epochs += 1

        if epoch >= MIN_EPOCHS_BEFORE_STOP and bad_epochs >= EARLY_STOP_PATIENCE:
            print(f"Early stopping: no VAL score improvement for {EARLY_STOP_PATIENCE} epochs.")
            break

    print(f"\nBest epoch: {best_epoch} | best VAL score: {best_score:.6f}")

    # ---- final test using best checkpoint ----
    ckpt = torch.load(best_path, map_location=device)
    memory.load_state_dict(ckpt["memory"])
    embedder.load_state_dict(ckpt["embedder"])
    classifier.load_state_dict(ckpt["classifier"])

    _ = eval_stream(train_data, test_data, stream_name="TEST(best)")

    cfg_path = artifacts / f"tgn_{MODEL_VARIANT}_config.json"
    cfg_path.write_text(
        json.dumps(
            ckpt["config"] | {"best_epoch": best_epoch, "best_val_score": best_score},
            indent=2,
        )
    )
    print("Saved config:", cfg_path)


if __name__ == "__main__":
    main()
