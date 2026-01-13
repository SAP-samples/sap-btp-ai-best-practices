from pathlib import Path


# We locate API root three levels up from api/app/regressor/.
API_ROOT = Path(__file__).resolve().parents[2]
# Repo root is two more levels up (sales-forecast-dashboard/api -> company_dashboard)
REPO_ROOT = API_ROOT.parents[1]
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = API_ROOT / "artifacts"


def ensure_artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def ensure_artifacts_subdir(name: str) -> Path:
    base = ensure_artifacts_dir()
    sub = base / name
    sub.mkdir(parents=True, exist_ok=True)
    return sub
