# Fraud Detection with Temporal Graph Features (PaySim)

Portfolio project: build a fraud scoring pipeline on a PaySim-style mobile money dataset and push beyond tabular baselines into **temporal + graph-based modeling**, including a streaming **Temporal Graph Network (TGN)**.

This repo is intentionally “research-y”: it emphasizes **time-respecting splits**, **rare-event evaluation**, and **leakage-aware feature design** more than squeezing a single leaderboard metric.

---

## Project status
✅ **Complete / archived** (learning-focused project).  
Key takeaway: on synthetic / constrained datasets, split design + leakage control often dominate model complexity, and a simpler model can be “good enough” once evaluation is realistic.

---

## What we built
A reproducible pipeline that can:

1) **Ingest PaySim transactions** (CSV/Parquet) and create a clean, time-sorted dataset  
2) Train a strong **tabular baseline** (LightGBM) with chronological splits  
3) Add **temporal + graph features** (velocity/recency, distinct counterparties, pair-history, degree spikes, simple cycle proxies)  
4) Train a **Temporal Graph Network (TGN)** for *event/edge fraud prediction* using streaming memory updates  
5) Evaluate like an operator: **recall/precision at fixed false-positive-rate (FPR) budgets** and optional **type gating** (e.g., CASH_OUT/TRANSFER)

---

## Why temporal graphs for fraud?
Fraud isn’t i.i.d.:
- accounts interact and evolve over time
- fraud rings create network signatures
- many signals are “within last X minutes/hours/days” (velocity/recency)

We model:
- **nodes** = accounts
- **edges** = timestamped transactions
- **edge label** = `isFraud`

---

## Dataset
PaySim-style dataset with columns like:  
`step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud`

**Data is not committed to the repo.** Put the raw CSV here:
- `data/raw/paysim.csv`

Common alternate PaySim filename also supported:
- `data/raw/PS_20174392719_1491204439457_log.csv`

Optional processed Parquet locations (used if present):
- `data/processed/paysim.parquet`
- `data/processed/paysim_with_temporal_graph_features.parquet`

---

## Modeling approach (high-level)

### 1) Tabular baseline (LightGBM)
- strict chronological train/val/test
- metrics include PR-AUC and operating points at fixed FPR budgets

### 2) Temporal + graph feature engineering
Features like:
- velocity / recency of activity (node-level)
- number of distinct counterparties over rolling windows
- pair-history features (orig→dest interaction recency/count)
- degree spikes / burstiness proxies

### 3) Streaming TGN (Temporal Graph Network)
The TGN learns node state over time:
- **Memory** per node updated after each event
- **Neighborhood aggregation** using a Transformer-style conv over the last neighbors
- **Event classifier** uses `(embedding(src), embedding(dst), edge message)` → fraud score

**Edge message variants**
- `behavioral`: `log1p(amount)` + transaction type one-hot  
- `mechanics`: decision-time-safe balances (e.g., `log1p(oldbalanceOrg)`, `log1p(oldbalanceDest)`)  
- `hybrid`: `behavioral + mechanics`

**Leakage control**
- Default is **decision-time-only mechanics** (no post-transaction deltas unless explicitly enabled)

---

## Evaluation (operator-style)
Because fraud is extremely rare, evaluation uses:
- **Recall & precision at fixed FPR budgets** (e.g., 0.1%–5%)
- Optional **type gating** (only consider alerts for `CASH_OUT` / `TRANSFER`)
- Time-based splitting modes, including a tail-based mode to reduce noisy validation when fraud counts are tiny

---

## Repository layout (typical)
- `src/gnn/` – temporal graph modeling (TGN) and utilities
- `src/features/` – temporal + graph feature generation (if present)
- `src/baselines/` – LightGBM baseline training (if present)
- `artifacts/` – saved configs / checkpoints / logs (not committed)

---

## Quickstart

### Install
```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

pip install -r requirements.txt
