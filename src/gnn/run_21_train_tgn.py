# src/gnn/run_21_train_tgn.py
# ------------------------------------------------------------
# TGN (Temporal Graph Network) for PaySim fraud event classification
# - Time-ordered training
# - VAL/TEST split can be "tail_fraud_with_min_test_events" (your current) OR "natural_tail"
# - MIN_VAL_FRAUD to reduce noisy validation
# - Edge message "msg" supports: behavioral | mechanics | hybrid (mechanics + behavior)
# - Optional decision-time-only mechanics (avoid post-transaction leakage)
# - Optional cold-start evaluation (unseen nodes in VAL/TEST)
#
# IMPORTANT FIXES INCLUDED:
# 1) LastNeighborLoader has NO .to(device) -> wrapped safely
# 2) e_id indexing MUST reference the same "history" arrays as the neighbor loader stream:
#    - training:   history = TRAIN only
#    - val scoring: history = TRAIN + VAL
#    - test scoring: history = TRAIN + VAL + TEST
# ------------------------------------------------------------

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyG TGN imports (with fallback for older PyG layouts)
try:
    from torch_geometric.nn.models.tgn import (
        TGNMemory,
        LastNeighborLoader,
        IdentityMessage,
        LastAggregator,
        TimeEncoder,
    )
except Exception:
    from torch_geometric.nn.models.tgn import TGNMemory, LastNeighborLoader, IdentityMessage, LastAggregator

    class TimeEncoder(nn.Module):
        def __init__(self, out_channels: int):
            super().__init__()
            self.out_channels = out_channels
            self.lin = nn.Linear(1, out_channels)

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            # t: [E] or [E,1]
            if t.dim() == 1:
                t = t.view(-1, 1)
            return torch.cos(self.lin(t.float()))

from torch_geometric.nn import TransformerConv

ColdStartMode = Literal["off", "either", "both"]
SplitMode = Literal["tail_fraud_with_min_test_events", "natural_tail"]


# -------------------------
# numerics + reproducibility
# -------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stable_sigmoid_np(x: np.ndarray) -> np.ndarray:
    # prevent exp overflow
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def make_last_neighbor_loader(num_nodes: int, size: int, device: torch.device):
    """
    PyG's LastNeighborLoader has NO .to(). It keeps its state internally (often CPU).
    We wrap it so outputs are moved to `device`, while inserts keep the internal state safe.

    The wrapper exposes:
      - __call__(seed_nodes) -> (n_id, edge_index, e_id) on `device`
      - insert(src, dst)     -> safely inserted (CPU-side state)
      - reset_state()
    """
    loader = LastNeighborLoader(num_nodes=num_nodes, size=size)

    class _Wrapped:
        def __init__(self, loader, device):
            self.loader = loader
            self.device = device

        def __call__(self, seed_nodes: torch.Tensor):
            n_id, edge_index, e_id = self.loader(seed_nodes.detach().cpu())
            return n_id.to(self.device), edge_index.to(self.device), e_id.to(self.device)

        def insert(self, src: torch.Tensor, dst: torch.Tensor):
            self.loader.insert(src.detach().cpu(), dst.detach().cpu())

        def reset_state(self):
            self.loader.reset_state()

    return _Wrapped(loader, device)


# -------------------------
# config + paths
# -------------------------
def repo_root() -> Path:
    # .../Fraud_Detection/src/gnn/run_21_train_tgn.py -> root is parents[2]
    return Path(__file__).resolve().parents[2]


def default_cfg() -> Dict:
    return {
        "MODEL_VARIANT": "behavioral",  # behavioral | mechanics | hybrid
        "SPLIT_MODE": "tail_fraud_with_min_test_events",  # or "natural_tail"

        "TRAIN_FRAC": 0.70,

        # ---- split knobs (tail-fraud mode) ----
        "TEST_FRAUD_TARGET": 100,
        "MIN_VAL_FRAUD": 80,
        "MIN_TEST_EVENTS": 10000,

        # ---- split knobs (natural tail mode) ----
        "TEST_EVENTS": 45000,

        # ---- evaluation knobs ----
        "ALLOWED_TYPES_GATE": ["CASH_OUT", "TRANSFER"],
        "SELECT_FPRS": [0.005, 0.01, 0.02, 0.05],
        "SELECT_WEIGHTS": [0.45, 0.25, 0.2, 0.1],

        # ---- model/training knobs ----
        "BATCH_SIZE": 2000,
        "EPOCHS": 25,
        "LR": 1e-3,
        "WEIGHT_DECAY": 1e-4,
        "CLIP_GRAD_NORM": 1.0,

        "USE_FOCAL": True,
        "FOCAL_GAMMA": 2.0,

        "MEMORY_DIM": 128,
        "TIME_DIM": 64,
        "HEADS": 2,
        "EMB_DIM": 128,
        "NEIGHBOR_SIZE": 50,

        # ---- realism knobs ----
        "DECISION_TIME_ONLY": True,   # mechanics features only use old balances (no newbalance/error deltas)
        "COLD_START_MODE": "off",     # off | either | both

        "EARLY_STOP_PATIENCE": 6,
        "MIN_EPOCHS_BEFORE_STOP": 3,
        "SEED": 42,
    }


def load_cfg(cfg_path: Optional[str]) -> Dict:
    cfg = default_cfg()
    if cfg_path:
        p = Path(cfg_path)
        if not p.is_absolute():
            p = repo_root() / p
        if p.exists():
            cfg.update(json.loads(p.read_text(encoding="utf-8")))
    return cfg


# -------------------------
# data loading
# -------------------------
def find_dataset_path() -> Path:
    root = repo_root()
    candidates = [
        root / "data" / "processed" / "paysim_with_temporal_graph_features.parquet",
        root / "data" / "processed" / "paysim.parquet",
        root / "data" / "raw" / "paysim.csv",
        root / "data" / "raw" / "PS_20174392719_1491204439457_log.csv",  # common PaySim filename
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not find PaySim dataset. Expected one of:\n" + "\n".join(str(x) for x in candidates)
    )


def load_paysim_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    required = ["step", "type", "amount", "nameOrig", "nameDest", "isFraud"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in dataset: {path}")

    df = df.copy()
    df["step"] = df["step"].astype(np.int64)
    df["isFraud"] = df["isFraud"].astype(np.int64)
    df["amount"] = df["amount"].astype(np.float32)

    for c in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        if c in df.columns:
            df[c] = df[c].astype(np.float32)

    df = df.sort_values("step").reset_index(drop=True)
    return df


def encode_nodes(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    all_nodes = pd.Index(df["nameOrig"]).append(pd.Index(df["nameDest"]))
    uniq = all_nodes.unique()
    node2id = {str(n): i for i, n in enumerate(uniq)}
    src = df["nameOrig"].astype(str).map(node2id).astype(np.int64).values
    dst = df["nameDest"].astype(str).map(node2id).astype(np.int64).values
    return src, dst, node2id


# -------------------------
# msg construction (behavioral / mechanics / hybrid)
# -------------------------
def build_edge_message_matrix(
    df: pd.DataFrame,
    model_variant: str,
    decision_time_only: bool = True,
) -> np.ndarray:
    """
    Returns msg matrix (N, D).
    - behavioral: log1p(amount) + type one-hot
    - mechanics: log1p(old balances) (+ optionally post balances & engineered errors if decision_time_only=False)
    - hybrid: concat(behavioral, mechanics)
    """
    amt = np.log1p(df["amount"].astype("float32").values).reshape(-1, 1)

    types = ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]
    t = df["type"].astype(str).values
    type_oh = np.stack([(t == k).astype("float32") for k in types], axis=1)
    behavioral = np.concatenate([amt, type_oh], axis=1).astype("float32")

    mech_cols = []
    for c in ["oldbalanceOrg", "oldbalanceDest"]:
        if c in df.columns:
            mech_cols.append(np.log1p(df[c].astype("float32").values).reshape(-1, 1))

    if not decision_time_only:
        for c in ["newbalanceOrig", "newbalanceDest"]:
            if c in df.columns:
                mech_cols.append(np.log1p(df[c].astype("float32").values).reshape(-1, 1))
        for c in ["orig_delta", "dest_delta", "orig_error", "dest_error", "abs_orig_error", "abs_dest_error"]:
            if c in df.columns:
                mech_cols.append(df[c].astype("float32").values.reshape(-1, 1))

    mechanics = (
        np.concatenate(mech_cols, axis=1).astype("float32")
        if mech_cols
        else np.zeros((len(df), 0), dtype="float32")
    )

    if model_variant == "behavioral":
        return behavioral
    if model_variant == "mechanics":
        return mechanics
    if model_variant == "hybrid":
        return np.concatenate([behavioral, mechanics], axis=1).astype("float32")

    raise ValueError(f"Unknown MODEL_VARIANT={model_variant}. Use behavioral|mechanics|hybrid")


# -------------------------
# splitting
# -------------------------
@dataclass
class SplitInfo:
    mode: str
    train_events: int
    val_events: int
    test_events: int
    train_fraud: int
    val_fraud: int
    test_fraud: int
    train_rate: float
    val_rate: float
    test_rate: float
    split_details: Dict


def _rate(n_pos: int, n: int) -> float:
    return float(n_pos) / float(n) if n else 0.0


def _apply_cold_start_filter(df: pd.DataFrame, train_nodes: set, mode: ColdStartMode) -> pd.DataFrame:
    if mode == "off":
        return df
    src_seen = df["nameOrig"].astype(str).isin(train_nodes)
    dst_seen = df["nameDest"].astype(str).isin(train_nodes)
    if mode == "both":
        keep = (~src_seen) & (~dst_seen)
    else:  # either
        keep = (~src_seen) | (~dst_seen)
    return df.loc[keep].copy()


def split_tail_fraud_with_min_test_events(
    df: pd.DataFrame,
    train_frac: float,
    test_fraud_target: int,
    min_val_fraud: int,
    min_test_events: int,
    cold_start_mode: ColdStartMode = "off",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitInfo]:
    """
    - TRAIN = first train_frac of timeline (fixed)
    - Tail = remaining
    - TEST = take from very end until reaching test_fraud_target fraud,
             but also enforce at least min_test_events rows.
    - VAL = the window right before TEST, expanded backward until >= min_val_fraud fraud.
    Optional cold-start applied to VAL/TEST after defining windows.
    """
    df = df.sort_values("step").reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    train_df = df.iloc[:n_train].copy()
    tail = df.iloc[n_train:].copy()

    y_tail = tail["isFraud"].values
    fraud_cum = (y_tail[::-1]).cumsum()
    hit = np.where(fraud_cum >= test_fraud_target)[0]
    if len(hit) == 0:
        take_rev = min(len(tail), max(min_test_events, 1))
        test_fraud_target_used = int(fraud_cum[-1]) if len(fraud_cum) else 0
        test_events_forced = True
    else:
        take_rev = hit[0] + 1
        test_fraud_target_used = test_fraud_target
        test_events_forced = False

    if take_rev < min_test_events:
        take_rev = min_test_events
        test_events_forced = True

    raw_test = tail.iloc[-take_rev:].copy()
    raw_val_pool = tail.iloc[:-take_rev].copy()

    train_nodes = set(train_df["nameOrig"].astype(str).unique()).union(
        set(train_df["nameDest"].astype(str).unique())
    )

    test_df = _apply_cold_start_filter(raw_test, train_nodes, cold_start_mode)
    val_pool = _apply_cold_start_filter(raw_val_pool, train_nodes, cold_start_mode)

    if min_val_fraud <= 0:
        val_df = val_pool.copy()
        val_forced_relax = False
    else:
        yv = val_pool["isFraud"].values
        fraud_cum_v = (yv[::-1]).cumsum()
        hitv = np.where(fraud_cum_v >= min_val_fraud)[0]
        if len(hitv) == 0:
            yv2 = raw_val_pool["isFraud"].values
            fraud_cum_v2 = (yv2[::-1]).cumsum()
            hitv2 = np.where(fraud_cum_v2 >= min_val_fraud)[0]
            if len(hitv2) == 0:
                raise ValueError(f"VAL cannot reach MIN_VAL_FRAUD={min_val_fraud} even without cold-start.")
            takev = hitv2[0] + 1
            val_df = raw_val_pool.iloc[-takev:].copy()
            val_forced_relax = True
        else:
            takev = hitv[0] + 1
            val_df = val_pool.iloc[-takev:].copy()
            val_forced_relax = False

    split_details = {
        "mode": "tail_fraud_with_min_test_events",
        "tail_fraud": int(tail["isFraud"].sum()),
        "test_fraud_target_used": int(test_fraud_target_used),
        "test_events_forced": bool(test_events_forced),
        "test_fraud_after_forcing": int(raw_test["isFraud"].sum()),
        "cold_start_mode": cold_start_mode,
        "val_relaxed_cold_start": bool(val_forced_relax),
    }

    info = SplitInfo(
        mode="tail_fraud_with_min_test_events",
        train_events=len(train_df),
        val_events=len(val_df),
        test_events=len(test_df),
        train_fraud=int(train_df["isFraud"].sum()),
        val_fraud=int(val_df["isFraud"].sum()),
        test_fraud=int(test_df["isFraud"].sum()),
        train_rate=_rate(int(train_df["isFraud"].sum()), len(train_df)),
        val_rate=_rate(int(val_df["isFraud"].sum()), len(val_df)),
        test_rate=_rate(int(test_df["isFraud"].sum()), len(test_df)),
        split_details=split_details,
    )
    return train_df, val_df, test_df, info


def split_natural_tail(
    df: pd.DataFrame,
    train_frac: float,
    test_events: int,
    min_val_fraud: int,
    cold_start_mode: ColdStartMode = "off",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitInfo]:
    """
    Honest split:
    - TRAIN = first train_frac
    - TEST = last test_events of remaining tail (natural base rate)
    - VAL = right before TEST, expand backward until >= min_val_fraud
    Optional cold-start for VAL/TEST.
    """
    df = df.sort_values("step").reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    train_df = df.iloc[:n_train].copy()
    tail = df.iloc[n_train:].copy()
    if len(tail) <= test_events + 1:
        raise ValueError(f"Not enough tail rows ({len(tail)}) for TEST_EVENTS={test_events}.")

    raw_test = tail.iloc[-test_events:].copy()
    raw_val_pool = tail.iloc[:-test_events].copy()

    train_nodes = set(train_df["nameOrig"].astype(str).unique()).union(
        set(train_df["nameDest"].astype(str).unique())
    )

    test_df = _apply_cold_start_filter(raw_test, train_nodes, cold_start_mode)
    val_pool = _apply_cold_start_filter(raw_val_pool, train_nodes, cold_start_mode)

    if min_val_fraud <= 0:
        val_df = val_pool.copy()
        val_forced_relax = False
    else:
        yv = val_pool["isFraud"].values
        fraud_cum = (yv[::-1]).cumsum()
        hit = np.where(fraud_cum >= min_val_fraud)[0]
        if len(hit) == 0:
            yv2 = raw_val_pool["isFraud"].values
            fraud_cum2 = (yv2[::-1]).cumsum()
            hit2 = np.where(fraud_cum2 >= min_val_fraud)[0]
            if len(hit2) == 0:
                raise ValueError(f"VAL cannot reach MIN_VAL_FRAUD={min_val_fraud} even without cold-start.")
            takev = hit2[0] + 1
            val_df = raw_val_pool.iloc[-takev:].copy()
            val_forced_relax = True
        else:
            takev = hit[0] + 1
            val_df = val_pool.iloc[-takev:].copy()
            val_forced_relax = False

    split_details = {
        "mode": "natural_tail",
        "test_events": int(test_events),
        "cold_start_mode": cold_start_mode,
        "val_relaxed_cold_start": bool(val_forced_relax),
    }

    info = SplitInfo(
        mode="natural_tail",
        train_events=len(train_df),
        val_events=len(val_df),
        test_events=len(test_df),
        train_fraud=int(train_df["isFraud"].sum()),
        val_fraud=int(val_df["isFraud"].sum()),
        test_fraud=int(test_df["isFraud"].sum()),
        train_rate=_rate(int(train_df["isFraud"].sum()), len(train_df)),
        val_rate=_rate(int(val_df["isFraud"].sum()), len(val_df)),
        test_rate=_rate(int(test_df["isFraud"].sum()), len(test_df)),
        split_details=split_details,
    )
    return train_df, val_df, test_df, info


# -------------------------
# model
# -------------------------
class TGNEncoder(nn.Module):
    """
    Memory -> neighborhood aggregation (TransformerConv) -> node embedding
    """
    def __init__(self, in_dim: int, emb_dim: int, heads: int, edge_dim: int, dropout: float = 0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.lin0 = nn.Linear(in_dim, emb_dim)
        self.conv1 = TransformerConv(
            in_channels=in_dim,
            out_channels=emb_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.lin = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # Safe fallback if subgraph has no edges:
        if edge_index is None or edge_index.numel() == 0 or edge_index.size(1) == 0:
            z = self.lin0(x)
            z = F.relu(z)
            return self.lin(z)

        z = self.conv1(x, edge_index, edge_attr)
        z = F.relu(z)
        z = self.lin(z)
        return z


class EventClassifier(nn.Module):
    def __init__(self, emb_dim: int, raw_msg_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * emb_dim + raw_msg_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_src, z_dst, msg], dim=1)
        return self.net(x).squeeze(-1)


# -------------------------
# losses
# -------------------------
def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    BCEWithLogits * focal factor.
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    focal = (1 - pt).pow(gamma)
    return (focal * bce).mean()


# -------------------------
# metrics @ FPR budgets
# -------------------------
def operating_points_at_fpr(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_fprs: List[float],
) -> List[Dict]:
    y_true = y_true.astype(np.int64)
    scores = scores.astype(np.float64)

    neg = (y_true == 0)
    n_neg = int(neg.sum())
    if n_neg == 0:
        return [{"target_fpr": f, "threshold": float("inf"), "k_flagged": 0, "recall": 0.0, "precision": 0.0} for f in target_fprs]

    order = np.argsort(-scores)  # descending
    y_sorted = y_true[order]
    s_sorted = scores[order]

    neg_sorted = (y_sorted == 0).astype(np.int64)
    pos_sorted = (y_sorted == 1).astype(np.int64)

    fp_cum = np.cumsum(neg_sorted)
    tp_cum = np.cumsum(pos_sorted)

    total_pos = int(pos_sorted.sum())

    out = []
    for fpr in target_fprs:
        fp_budget = fpr * n_neg
        ok = np.where(fp_cum <= fp_budget)[0]
        if len(ok) == 0:
            k = 0
        else:
            k = int(ok[-1] + 1)

        if k <= 0:
            thr = float("inf")
            tp = 0
            fp = 0
        else:
            thr = float(s_sorted[k - 1])
            tp = int(tp_cum[k - 1])
            fp = int(fp_cum[k - 1])

        recall = float(tp) / float(total_pos) if total_pos else 0.0
        precision = float(tp) / float(tp + fp) if (tp + fp) else 0.0

        out.append(
            {
                "target_fpr": float(fpr),
                "threshold": thr,
                "k_flagged": int(k),
                "recall": recall,
                "precision": precision,
            }
        )
    return out


def selection_score(op_points: List[Dict], fprs: List[float], weights: List[float]) -> float:
    r = {float(d["target_fpr"]): float(d["recall"]) for d in op_points}
    return float(sum(w * r.get(float(f), 0.0) for f, w in zip(fprs, weights)))


# -------------------------
# streaming batch iterator
# -------------------------
def batch_indices(n: int, batch_size: int):
    for i in range(0, n, batch_size):
        yield slice(i, min(i + batch_size, n))


# -------------------------
# main training loop helpers
# -------------------------
@torch.no_grad()
def init_memory_with_events(
    memory: TGNMemory,
    neighbor_loader: Any,
    src: torch.Tensor,
    dst: torch.Tensor,
    t: torch.Tensor,
    msg: torch.Tensor,
    batch_size: int,
):
    """
    "Warm up" memory/neighbors on a sequence of events without computing loss.
    """
    memory.reset_state()
    neighbor_loader.reset_state()

    n = src.numel()
    for sl in batch_indices(n, batch_size):
        s = src[sl]
        d = dst[sl]
        tt = t[sl]
        mm = msg[sl]
        neighbor_loader.insert(s, d)
        memory.update_state(s, d, tt, mm)
    memory.detach()


def build_embeddings_for_batch(
    memory: TGNMemory,
    neighbor_loader: Any,
    gnn: nn.Module,
    time_enc: nn.Module,
    assoc: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    t_now: torch.Tensor,
    t_hist: torch.Tensor,
    msg_hist: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute node embeddings for src/dst at time t_now using last-neighbor subgraph.

    CRITICAL: e_id indexes the *history stream since the last reset_state()*.
    Therefore, t_hist/msg_hist MUST be aligned with the same stream ordering.
    """
    n_query = torch.unique(torch.cat([src, dst], dim=0))
    n_id, edge_index, e_id = neighbor_loader(n_query)

    assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

    mem_x, _last_update = memory(n_id)

    # Use a single query time for the batch (safe + monotonic): max time in this batch
    t_query = t_now.max()

    if e_id.numel() == 0:
        # no historical edges yet
        z = gnn(mem_x, edge_index, torch.empty((0, msg_hist.size(1) + time_enc.out_channels), device=mem_x.device))
        z_src = z[assoc[src]]
        z_dst = z[assoc[dst]]
        return z_src, z_dst

    e_t = t_hist[e_id]
    dt = (t_query - e_t).clamp(min=0).to(torch.float32)
    dt_emb = time_enc(dt)

    e_msg = msg_hist[e_id]
    edge_attr = torch.cat([e_msg, dt_emb], dim=1)

    z = gnn(mem_x, edge_index, edge_attr)
    z_src = z[assoc[src]]
    z_dst = z[assoc[dst]]
    return z_src, z_dst


def train_one_epoch(
    memory: TGNMemory,
    neighbor_loader: Any,
    gnn: nn.Module,
    classifier: nn.Module,
    time_enc: nn.Module,
    assoc: torch.Tensor,
    opt: torch.optim.Optimizer,
    src: torch.Tensor,
    dst: torch.Tensor,
    t: torch.Tensor,
    msg: torch.Tensor,
    y: torch.Tensor,
    t_hist: torch.Tensor,
    msg_hist: torch.Tensor,
    batch_size: int,
    pos_weight: torch.Tensor,
    use_focal: bool,
    focal_gamma: float,
    clip_grad_norm: float,
) -> float:
    memory.train()
    gnn.train()
    classifier.train()
    time_enc.train()

    total_loss = 0.0
    n_batches = 0

    n = src.numel()
    for sl in batch_indices(n, batch_size):
        s = src[sl]
        d = dst[sl]
        tt = t[sl]
        mm = msg[sl]
        yy = y[sl].float()

        # predict BEFORE inserting current edges
        z_s, z_d = build_embeddings_for_batch(
            memory=memory,
            neighbor_loader=neighbor_loader,
            gnn=gnn,
            time_enc=time_enc,
            assoc=assoc,
            src=s,
            dst=d,
            t_now=tt,
            t_hist=t_hist,
            msg_hist=msg_hist,
        )
        logits = classifier(z_s, z_d, mm)

        if use_focal:
            loss = focal_bce_with_logits(logits, yy, pos_weight=pos_weight, gamma=focal_gamma)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, yy, pos_weight=pos_weight)

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if clip_grad_norm and clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                list(gnn.parameters()) + list(classifier.parameters()) + list(time_enc.parameters()),
                clip_grad_norm,
            )

        opt.step()

        # now update neighbor/memory with the current batch (label-free updates)
        neighbor_loader.insert(s, d)
        memory.update_state(s, d, tt, mm)
        memory.detach()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def score_stream(
    memory: TGNMemory,
    neighbor_loader: Any,
    gnn: nn.Module,
    classifier: nn.Module,
    time_enc: nn.Module,
    assoc: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    t: torch.Tensor,
    msg: torch.Tensor,
    t_hist: torch.Tensor,
    msg_hist: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    memory.eval()
    gnn.eval()
    classifier.eval()
    time_enc.eval()

    scores = np.zeros(src.numel(), dtype=np.float32)

    n = src.numel()
    idx0 = 0
    for sl in batch_indices(n, batch_size):
        s = src[sl]
        d = dst[sl]
        tt = t[sl]
        mm = msg[sl]

        z_s, z_d = build_embeddings_for_batch(
            memory=memory,
            neighbor_loader=neighbor_loader,
            gnn=gnn,
            time_enc=time_enc,
            assoc=assoc,
            src=s,
            dst=d,
            t_now=tt,
            t_hist=t_hist,
            msg_hist=msg_hist,
        )
        logits = classifier(z_s, z_d, mm).detach().cpu().numpy()
        probs = stable_sigmoid_np(logits)

        k = sl.stop - sl.start
        scores[idx0: idx0 + k] = probs
        idx0 += k

        # update streaming state AFTER scoring
        neighbor_loader.insert(s, d)
        memory.update_state(s, d, tt, mm)

    memory.detach()
    return scores


def evaluate_split(
    name: str,
    df_split: pd.DataFrame,
    y_true: np.ndarray,
    scores: np.ndarray,
    allowed_types_gate: List[str],
    target_fprs: List[float],
) -> Tuple[List[Dict], List[Dict]]:
    print(f"\n=== {name} RAW ===")
    print(f"rows: {len(df_split)} fraud: {int(y_true.sum())} rate: {float(y_true.mean()) if len(y_true) else 0.0}")

    op_raw = operating_points_at_fpr(y_true, scores, target_fprs)
    for d in op_raw:
        print(d)

    print(f"\n=== {name} GATED {allowed_types_gate} ===")
    allowed = df_split["type"].astype(str).isin(set(allowed_types_gate)).values
    scores_g = scores.copy()
    scores_g[~allowed] = -1e9
    op_g = operating_points_at_fpr(y_true, scores_g, target_fprs)
    for d in op_g:
        print(d)

    return op_raw, op_g


# -------------------------
# main
# -------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON (relative to repo root is ok)")
    args = parser.parse_args()

    CFG = load_cfg(args.config)
    set_seed(int(CFG["SEED"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Load data
    data_path = find_dataset_path()
    df = load_paysim_df(data_path)

    # Split
    split_mode: SplitMode = CFG.get("SPLIT_MODE", "tail_fraud_with_min_test_events")
    cold_start_mode: ColdStartMode = CFG.get("COLD_START_MODE", "off")

    if split_mode == "tail_fraud_with_min_test_events":
        train_df, val_df, test_df, split_info = split_tail_fraud_with_min_test_events(
            df=df,
            train_frac=float(CFG["TRAIN_FRAC"]),
            test_fraud_target=int(CFG["TEST_FRAUD_TARGET"]),
            min_val_fraud=int(CFG["MIN_VAL_FRAUD"]),
            min_test_events=int(CFG["MIN_TEST_EVENTS"]),
            cold_start_mode=cold_start_mode,
        )
        print("\n=== TIME SPLIT (TRAIN fixed, VAL/TEST tail by fraud) ===")
        print(
            f"TRAIN_FRAC={CFG['TRAIN_FRAC']:.2f} | TEST_FRAUD_TARGET={CFG['TEST_FRAUD_TARGET']} | "
            f"MIN_VAL_FRAUD={CFG['MIN_VAL_FRAUD']} | MIN_TEST_EVENTS={CFG['MIN_TEST_EVENTS']}"
        )
        print(f"split_info: {split_info.split_details}")
    else:
        train_df, val_df, test_df, split_info = split_natural_tail(
            df=df,
            train_frac=float(CFG["TRAIN_FRAC"]),
            test_events=int(CFG.get("TEST_EVENTS", 45000)),
            min_val_fraud=int(CFG.get("MIN_VAL_FRAUD", 80)),
            cold_start_mode=cold_start_mode,
        )
        print("\n=== TIME SPLIT (TRAIN fixed, VAL min-fraud, TEST natural tail) ===")
        print(
            f"TRAIN_FRAC={CFG['TRAIN_FRAC']:.2f} | TEST_EVENTS={CFG.get('TEST_EVENTS', 45000)} | "
            f"MIN_VAL_FRAUD={CFG.get('MIN_VAL_FRAUD', 80)}"
        )
        print(f"split_info: {split_info.split_details}")

    print(
        f"TRAIN: events={split_info.train_events} fraud={split_info.train_fraud} rate={split_info.train_rate:.6f}\n"
        f"VAL:   events={split_info.val_events} fraud={split_info.val_fraud} rate={split_info.val_rate:.6f}\n"
        f"TEST:  events={split_info.test_events} fraud={split_info.test_fraud} rate={split_info.test_rate:.6f}\n"
        "======================================================"
    )

    # Encode nodes globally on full df
    src_all, dst_all, node2id = encode_nodes(df)
    num_nodes = len(node2id)

    # Messages on full df
    model_variant = str(CFG["MODEL_VARIANT"])
    decision_time_only = bool(CFG.get("DECISION_TIME_ONLY", True))
    msg_all = build_edge_message_matrix(df, model_variant=model_variant, decision_time_only=decision_time_only)
    raw_msg_dim = msg_all.shape[1]

    # Labels + time on full df
    y_all = df["isFraud"].astype(np.int64).values
    t_all = df["step"].astype(np.int64).values

    # Indices (relative to df sorted+reset)
    train_idx = np.asarray(train_df.index.values, dtype=np.int64)
    val_idx = np.asarray(val_df.index.values, dtype=np.int64)
    test_idx = np.asarray(test_df.index.values, dtype=np.int64)

    # Torch tensors (full)
    src_t = torch.from_numpy(src_all).to(device)
    dst_t = torch.from_numpy(dst_all).to(device)
    t_t = torch.from_numpy(t_all).to(device)
    msg_t = torch.from_numpy(msg_all).to(device)
    y_t = torch.from_numpy(y_all).to(device)

    # Split tensors
    src_tr, dst_tr, t_tr, msg_tr, y_tr = src_t[train_idx], dst_t[train_idx], t_t[train_idx], msg_t[train_idx], y_t[train_idx]
    src_va, dst_va, t_va, msg_va, y_va = src_t[val_idx], dst_t[val_idx], t_t[val_idx], msg_t[val_idx], y_t[val_idx]
    src_te, dst_te, t_te, msg_te, y_te = src_t[test_idx], dst_t[test_idx], t_t[test_idx], msg_t[test_idx], y_t[test_idx]

    # pos_weight
    train_pos = int(y_tr.sum().item())
    train_neg = int(y_tr.numel() - train_pos)
    pos_weight_val = float(train_neg) / float(max(train_pos, 1))
    pos_weight = torch.tensor([pos_weight_val], device=device, dtype=torch.float32)

    print(f"\ntrain events: {y_tr.numel()}")
    print(f"train pos: {train_pos} neg: {train_neg} pos_weight: {pos_weight_val:.6f}")

    # Memory + neighbor loader (wrapped)
    memory = TGNMemory(
        num_nodes=num_nodes,
        raw_msg_dim=raw_msg_dim,
        memory_dim=int(CFG["MEMORY_DIM"]),
        time_dim=int(CFG["TIME_DIM"]),
        message_module=IdentityMessage(raw_msg_dim, int(CFG["MEMORY_DIM"]), int(CFG["TIME_DIM"])),
        aggregator_module=LastAggregator(),
    ).to(device)

    neighbor_loader = make_last_neighbor_loader(
        num_nodes=num_nodes,
        size=int(CFG["NEIGHBOR_SIZE"]),
        device=device,
    )

    # Time encoder for neighbor edge deltas
    time_enc = TimeEncoder(out_channels=int(CFG["TIME_DIM"])).to(device)

    # GNN edge_dim = raw_msg_dim + time_dim
    gnn = TGNEncoder(
        in_dim=int(CFG["MEMORY_DIM"]),
        emb_dim=int(CFG["EMB_DIM"]),
        heads=int(CFG["HEADS"]),
        edge_dim=raw_msg_dim + int(CFG["TIME_DIM"]),
        dropout=0.1,
    ).to(device)

    classifier = EventClassifier(
        emb_dim=int(CFG["EMB_DIM"]),
        raw_msg_dim=raw_msg_dim,
        hidden=256,
        dropout=0.1,
    ).to(device)

    params = list(gnn.parameters()) + list(classifier.parameters()) + list(time_enc.parameters()) + list(memory.parameters())
    opt = torch.optim.AdamW(params, lr=float(CFG["LR"]), weight_decay=float(CFG["WEIGHT_DECAY"]))

    assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

    # Artifacts
    art_dir = repo_root() / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    best_path = art_dir / f"tgn_{model_variant}.pt"
    cfg_out_path = art_dir / f"tgn_{model_variant}_config.json"

    # Early stopping
    best_val = -1e9
    best_epoch = -1
    no_improve = 0

    target_fprs = [0.001, 0.005, 0.01, 0.02, 0.05]
    select_fprs = [float(x) for x in CFG["SELECT_FPRS"]]
    select_w = [float(x) for x in CFG["SELECT_WEIGHTS"]]
    allowed_types = list(CFG.get("ALLOWED_TYPES_GATE", ["CASH_OUT", "TRANSFER"]))

    epochs = int(CFG["EPOCHS"])
    batch_size = int(CFG["BATCH_SIZE"])

    # Training epochs
    for epoch in range(1, epochs + 1):
        # Reset memory each epoch and replay TRAIN as streaming training
        memory.reset_state()
        neighbor_loader.reset_state()

        # Training history is TRAIN only (since we reset at epoch start)
        loss = train_one_epoch(
            memory=memory,
            neighbor_loader=neighbor_loader,
            gnn=gnn,
            classifier=classifier,
            time_enc=time_enc,
            assoc=assoc,
            opt=opt,
            src=src_tr,
            dst=dst_tr,
            t=t_tr,
            msg=msg_tr,
            y=y_tr,
            t_hist=t_tr,
            msg_hist=msg_tr,
            batch_size=batch_size,
            pos_weight=pos_weight,
            use_focal=bool(CFG["USE_FOCAL"]),
            focal_gamma=float(CFG["FOCAL_GAMMA"]),
            clip_grad_norm=float(CFG["CLIP_GRAD_NORM"]),
        )
        print(f"epoch {epoch}/{epochs} train_loss: {loss:.6f}")

        # --- Evaluation: warm-up on TRAIN, then score VAL with history TRAIN+VAL ---
        init_memory_with_events(memory, neighbor_loader, src_tr, dst_tr, t_tr, msg_tr, batch_size=batch_size)

        t_hist_val = torch.cat([t_tr, t_va], dim=0)
        msg_hist_val = torch.cat([msg_tr, msg_va], dim=0)

        val_scores = score_stream(
            memory=memory,
            neighbor_loader=neighbor_loader,
            gnn=gnn,
            classifier=classifier,
            time_enc=time_enc,
            assoc=assoc,
            src=src_va,
            dst=dst_va,
            t=t_va,
            msg=msg_va,
            t_hist=t_hist_val,
            msg_hist=msg_hist_val,
            batch_size=batch_size,
        )

        op_raw_val, _op_g_val = evaluate_split(
            name=f"VAL(epoch={epoch})",
            df_split=val_df,
            y_true=y_va.detach().cpu().numpy(),
            scores=val_scores,
            allowed_types_gate=allowed_types,
            target_fprs=target_fprs,
        )

        val_score = selection_score(op_raw_val, select_fprs, select_w)
        print(f"VAL selection score (mix @ {tuple(select_fprs)}): {val_score:.6f}")

        improved = val_score > best_val + 1e-12
        if improved:
            best_val = val_score
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "memory": memory.state_dict(),
                    "gnn": gnn.state_dict(),
                    "classifier": classifier.state_dict(),
                    "time_enc": time_enc.state_dict(),
                    "cfg": CFG,
                    "split_info": asdict(split_info),
                    "raw_msg_dim": raw_msg_dim,
                    "num_nodes": num_nodes,
                },
                best_path,
            )
            print(f"saved best by VAL score: {best_path}")
        else:
            no_improve += 1

        if epoch >= int(CFG["MIN_EPOCHS_BEFORE_STOP"]) and no_improve >= int(CFG["EARLY_STOP_PATIENCE"]):
            print(f"Early stopping: no VAL score improvement for {no_improve} epochs.")
            break

    print(f"\nBest epoch: {best_epoch} | best VAL score: {best_val:.6f}")

    # Load best
    ckpt = torch.load(best_path, map_location=device)
    memory.load_state_dict(ckpt["memory"])
    gnn.load_state_dict(ckpt["gnn"])
    classifier.load_state_dict(ckpt["classifier"])
    time_enc.load_state_dict(ckpt["time_enc"])

    # --- Final TEST scoring: warm-up on TRAIN, stream VAL (update), then stream TEST (score) ---
    init_memory_with_events(memory, neighbor_loader, src_tr, dst_tr, t_tr, msg_tr, batch_size=batch_size)

    t_hist_all = torch.cat([t_tr, t_va, t_te], dim=0)
    msg_hist_all = torch.cat([msg_tr, msg_va, msg_te], dim=0)

    # stream through VAL to put memory at "just before test"
    _ = score_stream(
        memory=memory,
        neighbor_loader=neighbor_loader,
        gnn=gnn,
        classifier=classifier,
        time_enc=time_enc,
        assoc=assoc,
        src=src_va,
        dst=dst_va,
        t=t_va,
        msg=msg_va,
        t_hist=t_hist_all,
        msg_hist=msg_hist_all,
        batch_size=batch_size,
    )

    test_scores = score_stream(
        memory=memory,
        neighbor_loader=neighbor_loader,
        gnn=gnn,
        classifier=classifier,
        time_enc=time_enc,
        assoc=assoc,
        src=src_te,
        dst=dst_te,
        t=t_te,
        msg=msg_te,
        t_hist=t_hist_all,
        msg_hist=msg_hist_all,
        batch_size=batch_size,
    )

    _op_raw_te, _op_g_te = evaluate_split(
        name="TEST(best)",
        df_split=test_df,
        y_true=y_te.detach().cpu().numpy(),
        scores=test_scores,
        allowed_types_gate=allowed_types,
        target_fprs=target_fprs,
    )

    # Save config + summary
    out_cfg = dict(CFG)
    out_cfg["split_info"] = asdict(split_info)
    out_cfg["best_epoch"] = int(best_epoch)
    out_cfg["best_val_score"] = float(best_val)
    out_cfg["raw_msg_dim"] = int(raw_msg_dim)
    out_cfg["num_nodes"] = int(num_nodes)
    out_cfg["data_path"] = str(data_path)

    cfg_out_path.write_text(json.dumps(out_cfg, indent=2), encoding="utf-8")
    print(f"Saved config: {cfg_out_path}")


if __name__ == "__main__":
    main()
