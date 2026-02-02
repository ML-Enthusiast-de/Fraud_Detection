from __future__ import annotations

from pathlib import Path
import json
import math
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
MODEL_VARIANT = "behavioral"   # artifact prefix; rename to "hybrid" if you want

BATCH_SIZE = 2000
EPOCHS = 15
LR = 1e-3

MEMORY_DIM = 128
TIME_DIM = 64
HEADS = 2
EMB_DIM = 128  # must be divisible by HEADS

NEIGHBOR_SIZE = 50  # recommended default; 100 can dilute top-of-list on small positives

# Loss / optimization helpers
CLIP_GRAD_NORM = 1.0

# Selection metric for checkpointing (rank-based, matches fraud ops)
# Weighted recall at fixed FPR budgets on VAL
SELECT_FPRS = (0.005, 0.01, 0.02)           # 0.5%, 1%, 2%
SELECT_WEIGHTS = (0.5, 0.3, 0.2)            # must sum to 1 ideally
EARLY_STOP_PATIENCE = 4                     # stop after N epochs w/o val score improvement
MIN_EPOCHS_BEFORE_STOP = 3                  # don't stop too early

# Optional ops-style gate for reporting (does NOT affect training)
ALLOWED_TYPES_GATE = {"TRANSFER", "CASH_OUT"}

# Reproducibility (best-effort)
SEED = 42
# =========================


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def time_split(num_events: int):
    i1 = int(0.70 * num_events)
    i2 = int(0.85 * num_events)
    return i1, i2


@torch.no_grad()
def rank_recall_at_fpr(y_true: np.ndarray, score: np.ndarray, target_fpr: float):
    """Rank-based operating point: choose top-k such that FPR <= target_fpr and recall is maximized."""
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
    i1, i2 = time_split(n_events)
    train_data = data[:i1]
    val_data = data[i1:i2]
    test_data = data[i2:]

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

    # Keep last-K neighbors per node
    neighbor_loader = LastNeighborLoader(num_nodes=num_nodes, size=NEIGHBOR_SIZE, device=device)

    # Neighborhood GNN edge_attr = [raw_msg, time_enc(age)]
    edge_dim = msg_dim + TIME_DIM
    embedder = TemporalGraphEmbedding(MEMORY_DIM, EMB_DIM, edge_dim=edge_dim, heads=HEADS).to(device)
    classifier = EdgeClassifier(EMB_DIM, msg_dim).to(device)

    params = list(memory.parameters()) + list(embedder.parameters()) + list(classifier.parameters())
    opt = torch.optim.Adam(params, lr=LR)

    # ---- Class imbalance ----
    y_train = train_data.y.cpu().numpy()
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32, device=device)

    print("train events:", len(train_data))
    print("train pos:", n_pos, "neg:", n_neg, "pos_weight:", float(pos_weight.item()))

    train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
    val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)

    # Node id -> local index mapping in sampled subgraph
    assoc = torch.empty(num_nodes, device=device, dtype=torch.long)

    def reset_state():
        memory.reset_state()
        neighbor_loader.reset_state()

    def batch_predict(batch: TemporalData, t_now: torch.Tensor):
        """
        Predict fraud logits for current batch WITHOUT updating memory with current batch.
        Uses neighborhood graph built from already-inserted edges.
        """
        # Seed nodes we want neighborhoods for
        seed = torch.cat([batch.src, batch.dst]).unique()

        # PyG signature: __call__(n_id)
        n_id, edge_index, e_id = neighbor_loader(seed)

        # Current memories for nodes in the sampled subgraph
        mem, _ = memory(n_id)

        # If there are no neighbor edges yet, skip convolution and just use memory
        if edge_index.numel() == 0 or e_id.numel() == 0:
            z = mem
        else:
            # Build edge_attr for neighborhood edges: [msg(edge), time_enc(age)]
            e_t = batch.t_all[e_id]  # timestamps of stored edges
            age = (t_now - e_t).clamp_min(0).to(mem.dtype)
            age_enc = memory.time_enc(age)            # [E, TIME_DIM]
            e_msg = batch.msg_all[e_id]               # [E, msg_dim]
            e_attr = torch.cat([e_msg, age_enc], dim=-1)

            z = embedder(mem, edge_index, e_attr)

        # Map global node ids -> local row indices in z
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z_src = z[assoc[batch.src]]
        z_dst = z[assoc[batch.dst]]

        logits = classifier(z_src, z_dst, batch.msg)
        return logits

    def train_one_epoch(epoch: int):
        memory.train()
        embedder.train()
        classifier.train()
        reset_state()

        # For training, e_id refers only to train inserts, so train-only arrays are fine
        train_t_all = train_data.t.to(device)
        train_msg_all = train_data.msg.to(device)

        total_loss = 0.0
        total = 0

        for batch in train_loader:
            batch = batch.to(device)
            batch.t_all = train_t_all
            batch.msg_all = train_msg_all

            opt.zero_grad(set_to_none=True)

            # Predict BEFORE state update (no leakage)
            t_now = batch.t.max()
            logits = batch_predict(batch, t_now)

            loss = F.binary_cross_entropy_with_logits(
                logits, batch.y.float(), pos_weight=pos_weight
            )
            loss.backward()

            if CLIP_GRAD_NORM is not None and CLIP_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(params, CLIP_GRAD_NORM)

            opt.step()

            # Update memory + neighbor store AFTER scoring
            memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            neighbor_loader.insert(batch.src, batch.dst)

            # Detach to stop gradients through time
            memory.detach()

            bs = batch.num_events
            total_loss += float(loss.item()) * bs
            total += bs

        return total_loss / max(1, total)

    @torch.no_grad()
    def eval_stream(history: TemporalData, stream: TemporalData, stream_name: str):
        """
        Warm-up on history (build memory+neighbors), then score stream sequentially.

        IMPORTANT: neighbor_loader keeps e_id in insertion order across history+stream,
        so we must index e_id into combined arrays all_t/all_msg = cat(history, stream).
        """
        memory.eval()
        embedder.eval()
        classifier.eval()
        reset_state()

        all_t = torch.cat([history.t, stream.t]).to(device)
        all_msg = torch.cat([history.msg, stream.msg]).to(device)

        # Warm-up: feed history only
        history_loader = TemporalDataLoader(history, batch_size=BATCH_SIZE)
        for batch in history_loader:
            batch = batch.to(device)
            memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            neighbor_loader.insert(batch.src, batch.dst)

        # Score stream sequentially
        stream_loader = TemporalDataLoader(stream, batch_size=BATCH_SIZE)

        all_logits = []
        all_y = []
        all_type = []

        for batch in stream_loader:
            batch = batch.to(device)
            batch.t_all = all_t
            batch.msg_all = all_msg

            t_now = batch.t.max()
            logits = batch_predict(batch, t_now)

            all_logits.append(logits.detach().cpu())
            all_y.append(batch.y.detach().cpu())
            all_type.append(batch.type_id.detach().cpu())

            # Update AFTER scoring
            memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            neighbor_loader.insert(batch.src, batch.dst)

        logits = torch.cat(all_logits).numpy()
        y_true = torch.cat(all_y).numpy().astype(np.int64)
        type_ids = torch.cat(all_type).numpy().astype(np.int64)
        score = 1.0 / (1.0 + np.exp(-logits))

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

        return {
            "rows": int(len(y_true)),
            "fraud": int(y_true.sum()),
            "fraud_rate": float(y_true.mean()),
            "raw_recalls": raw_recalls,
            "gated_recalls": gated_recalls,
        }

    # ---- checkpoint selection ----
    best_path = artifacts / f"tgn_{MODEL_VARIANT}.pt"
    best_score = -1.0
    best_epoch = -1
    bad_epochs = 0

    # Save config (once; we’ll also re-save inside checkpoint for reproducibility)
    base_config = {
        "MODEL_VARIANT": MODEL_VARIANT,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "MEMORY_DIM": MEMORY_DIM,
        "TIME_DIM": TIME_DIM,
        "HEADS": HEADS,
        "EMB_DIM": EMB_DIM,
        "NEIGHBOR_SIZE": NEIGHBOR_SIZE,
        "ALLOWED_TYPES_GATE": sorted(ALLOWED_TYPES_GATE),
        "SELECT_FPRS": list(SELECT_FPRS),
        "SELECT_WEIGHTS": list(SELECT_WEIGHTS),
        "EARLY_STOP_PATIENCE": EARLY_STOP_PATIENCE,
        "SEED": SEED,
        "CLIP_GRAD_NORM": CLIP_GRAD_NORM,
    }

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(epoch)
        print(f"epoch {epoch}/{EPOCHS} train_loss: {loss:.6f}")

        val_out = eval_stream(train_data, val_data, stream_name=f"VAL(epoch={epoch})")

        # Selection score: weighted average of raw recalls at selected FPRs
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

        # Early stopping
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
    cfg_path.write_text(json.dumps(ckpt["config"] | {"best_epoch": best_epoch, "best_val_score": best_score}, indent=2))
    print("Saved config:", cfg_path)


if __name__ == "__main__":
    main()
