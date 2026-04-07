"""Project paths and hyperparameters."""

from pathlib import Path

# Repo root (parent of ``src/``) — paths work no matter where notebooks set cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = str(PROJECT_ROOT / "data" / "raw")
DATA_FILTERED_DIR = str(PROJECT_ROOT / "data" / "filtered")
DATA_PROCESSED_DIR = str(PROJECT_ROOT / "data" / "processed")
FIGURES_DIR = str(PROJECT_ROOT / "outputs" / "figures")
POSTERIORS_DIR = str(PROJECT_ROOT / "outputs" / "posteriors")
QUALITY_REPORT_DIR = str(PROJECT_ROOT / "outputs" / "quality_report")

RANDOM_SEED = 42
TEST_SIZE = 0.2
N_BINS_ECE = 10

MCMC_DRAWS = 2000
MCMC_TUNE = 1000
MCMC_CHAINS = 4
MCMC_TARGET_ACCEPT = 0.90

EPSILON = 1e-4
MIN_CATEGORY_SIZE = 20

# ── Data Quality Thresholds (edit these after inspecting your data) ──
MIN_VOLUME_USD = 1000  # minimum total trading volume in USD
MIN_LIFETIME_DAYS = 7  # minimum days market must have existed
STUCK_PROB_LOW = 0.02  # prices below this are considered stuck
STUCK_PROB_HIGH = 0.98  # prices above this are considered stuck
MIN_LIQUIDITY_USD = 0  # set > 0 if you want to filter on liquidity too
