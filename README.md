PROJECT: Hierarchical Bayesian Recalibration of Prediction Market Probabilities on Polymarket
COURSE: CSC4850 Advanced Machine Learning, Spring 2026, Georgia State University
TEAM: Yatharth Gohil, Jinash Rouniyar

GOAL: Build a complete machine learning pipeline that:
  1. Loads the raw Polymarket Kaggle dataset
  2. Applies rigorous data quality filtering to produce a clean, analysis-ready dataset
  3. Performs EDA on the CLEAN dataset only (never on raw data)
  4. Fits a baseline logistic recalibration model
  5. Fits a Hierarchical Bayesian Recalibration Model (HBRM) using PyMC
  6. Evaluates all models using Brier score, log loss, ECE, and reliability diagrams
  7. Produces publication-ready figures for an IEEE-format paper and class presentation

  DATA QUALITY PHILOSOPHY:
  The Polymarket dataset contains many markets that are NOT suitable for calibration analysis:
  - Ghost markets with near-zero volume (one trader, not crowd wisdom)
  - Prices stuck at 0.01 or 0.99 (never traded, initialization artifact)
  - Cancelled or ambiguously resolved markets
  - Markets that existed for less than 7 days (no time to aggregate information)
  - Categories with fewer than 20 markets (too few for hierarchical estimation)
  ALL of these must be removed BEFORE any analysis. The filtering step is itself a
  reportable result — include a data filtering table in the IEEE paper.

DATASET: Polymarket Prediction Markets from Kaggle
  - URL: https://www.kaggle.com/datasets/ismetsemedov/polymarket-prediction-markets
  - File will be at: data/raw/ (user will place CSV there)
  - Key columns: market-implied YES probability, binary outcome (YES/NO resolved),
    category (politics/crypto/sports/etc.), volume, liquidity, spread, timestamps

ENVIRONMENT:
  - Python 3.12
  - PyMC installed via conda-forge: conda install -c conda-forge pymc
  - All other packages via pip: arviz scikit-learn matplotlib seaborn pandas numpy jupyterlab

FOLDER STRUCTURE (already created or create it):
  polymarket_hbrm/
  ├── data/
  │   ├── raw/          ← user places Kaggle CSV here (NEVER read directly into models)
  │   ├── filtered/     ← output of quality filtering step (clean dataset lives here)
  │   └── processed/    ← train/test splits from the filtered dataset
  ├── notebooks/
  ├── src/
  │   ├── preprocess.py   ← parsing functions
  │   ├── quality.py      ← NEW: data quality filtering functions
  │   ├── models.py
  │   └── evaluate.py
  └── outputs/
      ├── figures/
      ├── posteriors/
      └── quality_report/ ← filtering summary tables saved here

THE CORE MODEL MATH:
  For each resolved binary market i in category j:
    logit(P(Y_i = 1)) = α_j + β_j * log_odds(market_prob_i)
                        + γ_vol * volume_z_i
                        + γ_liq * liquidity_z_i
                        + γ_spr * spread_z_i
    Y_i ~ Bernoulli(sigmoid(logit_p_i))

  Priors:
    μ_α ~ Normal(0, 1)           # global intercept hyperprior
    μ_β ~ Normal(1, 0.5)         # global slope hyperprior (1 = perfect calibration)
    σ_α ~ HalfNormal(0.5)        # between-category intercept variation
    σ_β ~ HalfNormal(0.5)        # between-category slope variation
    α_j ~ Normal(μ_α, σ_α)       # category-level intercept (non-centered)
    β_j ~ Normal(μ_β, σ_β)       # category-level slope (non-centered)
    γ_* ~ Normal(0, 0.5)         # covariate effects

INTERPRETATION OF PARAMETERS:
  α_j > 0: category j systematically underestimates YES probability
  α_j < 0: category j systematically overestimates YES probability
  β_j > 1: market is underconfident (spreads too wide)
  β_j < 1: market is overconfident (too extreme in its probabilities)
  β_j = 1 and α_j = 0: perfectly calibrated


# Environment Setup

## 1) Create and activate the conda environment

```bash
conda create -n hbrm python=3.12 -y
conda activate hbrm
```

## 2) Install PyMC from conda-forge

```bash
conda install -c conda-forge pymc -y
```

## 3) Install C/C++ toolchain (recommended for fast PyMC sampling)

Without a compiler, PyTensor falls back to pure Python and MCMC becomes much slower.

```bash
conda install -c conda-forge cxx-compiler gxx -y
```

## 4) Install remaining Python dependencies

```bash
pip install -r requirements.txt
```

## 5) Register Jupyter kernel (so notebooks use this env)

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name hbrm --display-name "Python (hbrm)"
```

## 6) Launch Jupyter

From the project root folder:

```bash
jupyter lab
```

In the notebook UI, select kernel: **Python (hbrm)**.

## 7) Quick verification

Run these in the activated `hbrm` environment:

```bash
python -c "import pandas, numpy, sklearn, pymc, arviz; print('Environment OK')"
python -c "import pytensor; print('PyTensor OK')"
```

If you still see a `g++ not detected` warning after installing compilers, restart your terminal, restart Cursor/Jupyter, and restart the notebook kernel.
