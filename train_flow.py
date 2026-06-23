import mlflow
import pandas as pd
from pathlib import Path

_HERE = Path(__file__).parent

from training import train
from feature_selection import build_config_from_features
from selected_feature_sets import (
    catboost_restricted,
)
from generate_dummy_data import generate_dummy_training_data


def run_single_training(
    config: dict,
    model_type: str = "EBM",
    experiment_name: str = "portfolio_forecast",
    start=pd.Timestamp(year=2022, month=2, day=1, tz="CET"),
    end=pd.Timestamp(year=2026, month=1, day=1, tz="CET"),
    additional_tag: dict = None,
    only_evaluate_on_test_set: bool = False,
):
    mlflow.set_tracking_uri(uri="http://localhost:5000")
    if not (mlflow.get_experiment_by_name(experiment_name)):
        mlflow.create_experiment(experiment_name)

    exp = mlflow.set_experiment(experiment_name=experiment_name)

    print("Generating dummy data to mimic data loading...")
    training_data = generate_dummy_training_data()
    training_data = training_data.loc[training_data.index >= start]
    training_data = training_data.loc[training_data.index <= end]

    # Ensure config is a single parameter set (not a list/grid)
    if isinstance(config, dict):
        param_set = {k: v[0] if isinstance(v, list) else v for k, v in config.items()}
    else:
        param_set = config

    train(
        exp,
        model_type,
        param_set,
        training_data,
        additional_tag=additional_tag,
        only_final_model=only_evaluate_on_test_set,
    )


if __name__ == "__main__":
    config = build_config_from_features(catboost_restricted, target="prod_act_sum")
    run_single_training(
        config,
        model_type="CatBoost",
        experiment_name="performance_comparison",
    )
