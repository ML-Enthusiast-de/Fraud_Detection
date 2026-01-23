import sys
print("PYTHON:", sys.executable)
print("VERSION:", sys.version)


from pathlib import Path
import sys

# --- make repo root importable no matter how the script is launched ---
REPO_ROOT = Path(__file__).resolve().parents[1]   # .../Fraud_Detection
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_DIR))
# ---------------------------------------------------------------------

from data.download_paysim import download_paysim
from data.load_paysim import load_and_profile


if __name__ == "__main__":
    data_path = REPO_ROOT / "data" / "raw" / "paysim.parquet"

    if not data_path.exists():
        print("Data not found -> downloading...")
        download_paysim(data_path, sample_rows=300_000)

    load_and_profile(data_path)
