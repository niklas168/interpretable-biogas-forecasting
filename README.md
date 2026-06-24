# interpretable-biogas-forecasting

Interpretable ML for biogas portfolio time series forecasting. Compares EBMs, GAMs, GAMINets, IGANN, CatBoost, XGBoost, Random Forest, MLP, SVM, and Prophet using MLflow for experiment tracking and Optuna for Bayesian hyperparameter tuning.

## Installation

Requires Python 3.11 and [Poetry](https://python-poetry.org/docs/#installation).

```bash
git clone <repo-url>
cd interpretable-biogas-forecasting
poetry install
```

> Some dependencies (EXNNs, NAM) are installed directly from GitHub and require network access during install.

## Setup

### 1. Generate dummy training data

Data is generated on-the-fly by the scripts. To manually run the data generation script:

```bash
poetry run python generate_dummy_data.py
```

This verifies the generation of a 34 321-row, 245-column DataFrame with realistic synthetic biogas portfolio features.

### 2. Start MLflow tracking server

All scripts connect to MLflow at `http://localhost:5000`. Start the server before running any training:

```bash
poetry run mlflow server --host 127.0.0.1 --port 5000
```

Open the UI at [http://localhost:5000](http://localhost:5000).

## Scripts

| Script | Description |
|---|---|
| `train_flow.py` | Train a single model and log results to MLflow. Edit the bottom of the file to choose model type, feature set, and experiment name. |
| `bayesian_tuning.py` | Run Optuna Bayesian hyperparameter search. Trials are persisted in `optuna_studies.db` (SQLite) so runs can be interrupted and resumed. |
| `benchmark.py` | Evaluate a naive benchmark column (e.g. a production forecast lag) against the target and log RMSE to MLflow. |
| `generate_dummy_data.py` | Generate synthetic dummy data. |
| `feature_selection.py` | Utilities for building feature config dicts from named feature sets. |
| `selected_feature_sets.py` | Predefined feature sets (e.g. `catboost_restricted`). |
| `models.py` | Model class definitions and the `_model_mapping` registry. |
| `training.py` | Core training logic: cross-validation, scaling, final model fitting, and MLflow logging. |
| `preprocessing_utils.py` | Feature subsetting, scaling helpers, and time feature engineering. |

