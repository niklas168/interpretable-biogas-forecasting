import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def run_benchmark(
    exp,
    training_data: pd.DataFrame,
    target_col: str,
    benchmark_col: str,
    n_splits: int = 5,
    additional_tag: dict = None,
):
    split_ts = pd.Timestamp("2025-01-01", tz="CET")
    train_df = training_data.loc[training_data.index < split_ts]
    test_df = training_data.loc[training_data.index >= split_ts]

    if benchmark_col not in training_data.columns:
        raise ValueError(
            f"Benchmark column {benchmark_col} not found in training data."
        )

    with mlflow.start_run(experiment_id=exp.experiment_id):
        if additional_tag:
            mlflow.set_tags(additional_tag)

        mlflow.set_tag("model_type", "Benchmark")
        mlflow.set_tag("benchmark_column", benchmark_col)

        mlflow.log_params(
            {
                "target": target_col,
                "benchmark_col": benchmark_col,
                "train_start": train_df.index[0],
                "train_end": train_df.index[-1],
                "test_start": test_df.index[0] if not test_df.empty else None,
                "test_end": test_df.index[-1] if not test_df.empty else None,
            }
        )

        # Cross-validation on train set
        tscv = TimeSeriesSplit(n_splits=n_splits)
        rmses = []
        for train_index, val_index in tscv.split(train_df):
            val_fold = train_df.iloc[val_index]
            y_true = val_fold[target_col]
            y_pred = val_fold[benchmark_col]

            # Handle NaNs in both benchmark and target (e.g. if lags go out of bounds or target is missing)
            valid_mask = ~y_true.isna() & ~y_pred.isna()
            if valid_mask.any():
                rmse = root_mean_squared_error(y_true[valid_mask], y_pred[valid_mask])
                rmses.append(rmse)

        if rmses:
            rmse_cv_mean = np.mean(rmses)
            rmse_cv_var = np.var(rmses)
            mlflow.log_metric("rmse_cv_mean", rmse_cv_mean)
            mlflow.log_metric("rmse_cv_var", rmse_cv_var)

        # Evaluation on test set
        if not test_df.empty:
            y_true_test = test_df[target_col]
            y_pred_test = test_df[benchmark_col]

            valid_mask_test = ~y_true_test.isna() & ~y_pred_test.isna()
            if valid_mask_test.any():
                rmse_test = root_mean_squared_error(
                    y_true_test[valid_mask_test], y_pred_test[valid_mask_test]
                )
                mlflow.log_metric("rmse_final_model", rmse_test)


if __name__ == "__main__":
    training_data = pd.read_pickle("cached_training_data.pkl")

    mlflow.set_tracking_uri(uri="http://localhost:5000")
    if not mlflow.get_experiment_by_name("benchmark_runs"):
        mlflow.create_experiment("benchmark_runs")
    exp = mlflow.set_experiment(experiment_name="benchmark_runs")

    run_benchmark(
        exp,
        training_data,
        target_col="prod_act_sum",
        benchmark_col="prod_fc_sum_lag_0_hours",
        additional_tag={"Benchmark Type": "Production Forecast"},
    )
