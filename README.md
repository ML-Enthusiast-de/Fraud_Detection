# Fraud Detection with Temporal Graphs (PaySim)

Portfolio project: build an end-to-end fraud detection system that moves from a strong baseline to
graph + temporal modeling, with a deployable scoring API.

## What we’re building
A reproducible pipeline that:
1) Ingests PaySim transactions (CSV) and produces a clean Parquet dataset
2) Trains a baseline model (tabular)
3) Adds **temporal + graph features** (velocity, recency, degree spikes, distinct counterparties, simple cycle proxies)
4) (Optional advanced) Trains a **Temporal Graph Neural Network (TGN)** for edge fraud prediction
5) Serves a fraud scoring endpoint (FastAPI) and adds monitoring hooks

## Why temporal graphs for fraud?
Fraud is rarely i.i.d.:
- accounts interact over time
- fraud rings create network signatures
- many signals are “within last X minutes/hours/days” (velocity/recency)

We model:
- nodes = accounts
- edges = timestamped transactions
- edge label = isFraud

## Dataset
We use a PaySim-style dataset (mobile money transactions) with columns like:
`step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud`.

**Data is not committed to the repo.** Put the raw CSV at:
`data/raw/paysim.csv`

## Quickstart
### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Fraud_Detection

