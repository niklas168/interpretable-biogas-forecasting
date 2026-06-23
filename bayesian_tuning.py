import gc
import mlflow
from mlflow.exceptions import MlflowException
import optuna
from optuna.trial import TrialState
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
)
import pandas as pd
import numpy as np
import os
import pickle
import tensorflow as tf
from pathlib import Path

from feature_selection import build_config_from_features
from training import _model_train_mapping
from preprocessing_utils import create_feature_subset
from models import _model_mapping


_STUDY_DB_PATH = Path(__file__).resolve().parent / "optuna_studies.db"
OPTUNA_STORAGE_URL = f"sqlite:///{_STUDY_DB_PATH}"


def get_search_space(model_type, trial):
    if model_type == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
        }
    elif model_type == "CatBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "random_strength": trial.suggest_float("random_strength", 0, 10),
            "verbose": False,
        }
    elif model_type == "EBM":
        return {
            "interactions": trial.suggest_int("interactions", 0, 20),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "max_bins": trial.suggest_int("max_bins", 32, 512),
            "outer_bags": trial.suggest_int("outer_bags", 4, 16),
            "inner_bags": trial.suggest_int("inner_bags", 0, 8),
            "max_leafes": trial.suggest_int("max_leafes", 1, 20),
        }
    elif model_type == "LinearGAM":
        return {
            "lam": trial.suggest_float("lam", 1e-3, 1e3, log=True),
            "n_splines": trial.suggest_int("n_splines", 8, 30),
            "spline_order": trial.suggest_int("spline_order", 2, 4),
            "max_iter": trial.suggest_int("max_iter", 100, 300),
            "tol": trial.suggest_float("tol", 1e-5, 1e-3, log=True),
        }
    elif model_type == "GAMINet":
        lr_all = trial.suggest_float("lr_bp", 1e-5, 1e-2, log=True)
        return {
            "interact_num": trial.suggest_int("interact_num", 0, 50),
            "reg_clarity": trial.suggest_float("reg_clarity", 0.01, 1.0, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [100, 200, 500]),
            "lr_bp": [lr_all, lr_all, lr_all],
            "heredity": trial.suggest_categorical("heredity", [True, False]),
            "main_effect_epochs": trial.suggest_int("main_effect_epochs", 1000, 10000),
            "interaction_epochs": trial.suggest_int("interaction_epochs", 1000, 10000),
        }
    elif model_type == "IGANN":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 1000, 10000),
            "boost_rate": trial.suggest_float("boost_rate", 0.01, 1.0, log=True),
            "init_reg": trial.suggest_int("init_reg", 1, 10),
            "elm_scale": trial.suggest_float("elm_scale", 0.1, 10.0, log=True),
            "elm_alpha": trial.suggest_float("elm_alpha", 0.1, 10.0, log=True),
            "act": trial.suggest_categorical("act", ["elu", "relu"]),
            "early_stopping": trial.suggest_int("early_stopping", 10, 200),
        }
    elif model_type == "Prophet":
        return {
            "growth": trial.suggest_categorical("growth", ["linear", "flat"]),
            "changepoint_prior_scale": trial.suggest_float(
                "changepoint_prior_scale", 1e-3, 5e-1, log=True
            ),
            "seasonality_prior_scale": trial.suggest_float(
                "seasonality_prior_scale", 1e-2, 1e1, log=True
            ),
            "holidays_prior_scale": trial.suggest_float(
                "holidays_prior_scale", 1e-2, 1e1, log=True
            ),
            "seasonality_mode": trial.suggest_categorical(
                "seasonality_mode", ["additive", "multiplicative"]
            ),
            "n_changepoints": trial.suggest_int("n_changepoints", 5, 50),
            "changepoint_range": trial.suggest_float("changepoint_range", 0.6, 0.95),
        }
    elif model_type == "RandomForest":
        use_max_depth = trial.suggest_categorical("use_max_depth", [True, False])
        max_depth = trial.suggest_int("max_depth", 5, 40) if use_max_depth else None
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": max_depth,
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
    elif model_type == "MLP":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_layer_sizes = []
        for i in range(n_layers):
            hidden_layer_sizes.append(trial.suggest_int(f"n_units_l{i}", 10, 200))
        return {
            "hidden_layer_sizes": tuple(hidden_layer_sizes),
            "activation": trial.suggest_categorical(
                "activation", ["identity", "logistic", "tanh", "relu"]
            ),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-2, log=True
            ),
        }
    elif model_type == "SVM":
        return {
            "kernel": trial.suggest_categorical(
                "kernel", ["linear", "rbf", "poly", "sigmoid"]
            ),
            "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
    return {}


def objective(trial, model_type, base_param_set, training_data, exp_id, n_splits=5):
    # Get hyperparameters for this trial
    hyperparams = get_search_space(model_type, trial)

    # Merge with base param set (which contains target, covariates, etc.)
    param_set = {**base_param_set, **hyperparams}

    # Prepare data
    data, target_col = create_feature_subset(param_set, training_data)
    split_ts = pd.Timestamp("2025-01-01", tz="CET")
    train_df = data.loc[data.index < split_ts]
    test_df = data.loc[data.index >= split_ts]

    # Create training config
    training_config = _model_train_mapping[model_type](
        model_type=model_type,
        param_set=param_set,
        training_data=train_df,
        testing_data=test_df,
    )

    # Start MLflow nested run
    with mlflow.start_run(experiment_id=exp_id, nested=True):
        mlflow.log_params(training_config.data_params)
        mlflow.log_params(training_config.hyperparam_config.model_dump())
        mlflow.log_params({"scaler_x": training_config.scaler_type_x})
        mlflow.log_params({"scaler_y": training_config.scaler_type_y})

        try:
            # Cross-validation
            rmse_mean, rmse_var = training_config.cross_validate(n_splits=n_splits)

            mlflow.log_metric("rmse_cv_mean", rmse_mean)
            mlflow.log_metric("rmse_cv_var", rmse_var)

            # Fit on whole train set and evaluate on test set
            final_model, rmse_test = training_config.fit_and_evaluate_final_model()
            mlflow.log_metric("rmse_final_model", rmse_test)

            # Store extra info in trial for parent run access
            trial.set_user_attr("rmse_cv_var", rmse_var)
            trial.set_user_attr("rmse_final_model", rmse_test)

            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("trial_number", trial.number)

            return rmse_mean
        finally:
            # GAMINet (TF/Keras) leaks graph + optimizer state across trials;
            # clearing the session releases what Python's GC can't.
            tf.keras.backend.clear_session()
            gc.collect()


def run_bayesian_tuning(
    model_type: str,
    feature_set: dict,
    feature_set_name: str,
    training_data: pd.DataFrame,
    n_trials: int = 50,
    experiment_name: str = "bayesian_tuning",
    create_new_study: bool = False,
):
    """Run Optuna tuning with a persistent SQLite-backed study.

    `n_trials` is the **total** target number of completed trials for the
    `(model_type, feature_set_name)` study. On resume, only the missing trials
    are run (e.g. 4 already complete + n_trials=25 → 21 more). FAILED trials do
    not consume budget; they are retried on the next call.

    The parent MLflow run is persisted in `study.user_attrs["mlflow_parent_run_id"]`,
    so resumed nested trial runs land under the original parent and the parent's
    summary metrics reflect the whole study. If the persisted run was deleted,
    a new parent run is created and re-persisted.

    Set `create_new_study=True` to discard any existing study with the same
    name and start fresh. Destructive — prior trials for that study are gone.
    """
    mlflow.set_tracking_uri(uri="http://localhost:5000")
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    exp = mlflow.set_experiment(experiment_name=experiment_name)

    study_name = f"{model_type}__{feature_set_name}"
    if create_new_study:
        try:
            optuna.delete_study(study_name=study_name, storage=OPTUNA_STORAGE_URL)
            print(f"Deleted existing study '{study_name}' before re-creating.")
        except KeyError:
            pass  # no prior study with that name
    study = optuna.create_study(
        direction="minimize",
        storage=OPTUNA_STORAGE_URL,
        study_name=study_name,
        load_if_exists=not create_new_study,
    )

    # Base param set from feature set (excluding the nested lists that ParameterGrid uses)
    # Since feature_set is usually defined like in train_flow.py with [[]] for covariates
    base_param_set = {
        "target": (
            feature_set["target"][0]
            if isinstance(feature_set["target"], list)
            else feature_set["target"]
        ),
        "past_covariates": (
            feature_set["past_covariates"][0]
            if isinstance(feature_set["past_covariates"], list)
            else feature_set["past_covariates"]
        ),
        "future_covariates": (
            feature_set["future_covariates"][0]
            if isinstance(feature_set["future_covariates"], list)
            else feature_set["future_covariates"]
        ),
        "scaler": feature_set.get("scaler", [["std"]])[0],
        "cyclic_time_features": feature_set.get("cyclic_time_features", [[]])[0],
    }

    # TODO: when an Optuna pruner is added, count COMPLETE | PRUNED states.
    completed = sum(t.state == TrialState.COMPLETE for t in study.trials)
    remaining = max(0, n_trials - completed)
    print(
        f"Study '{study_name}': {completed}/{n_trials} complete → running {remaining} more."
    )

    existing_run_id = study.user_attrs.get("mlflow_parent_run_id")
    parent_run_cm = None
    if existing_run_id:
        try:
            parent_run_cm = mlflow.start_run(run_id=existing_run_id)
        except MlflowException as e:
            print(
                f"Could not resume MLflow run {existing_run_id} ({e}); starting a new parent run."
            )
            parent_run_cm = None
    if parent_run_cm is None:
        parent_run_cm = mlflow.start_run(experiment_id=exp.experiment_id)
        study.set_user_attr("mlflow_parent_run_id", parent_run_cm.info.run_id)

    with parent_run_cm:
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("feature_set", feature_set_name)

        if remaining > 0:
            study.optimize(
                lambda trial: objective(
                    trial, model_type, base_param_set, training_data, exp.experiment_id
                ),
                n_trials=remaining,
            )

        # Log Optuna plots
        try:
            fig_hist = plot_optimization_history(study)
            fig_hist.write_html("optimization_history.html")
            mlflow.log_artifact("optimization_history.html")
            os.remove("optimization_history.html")

            fig_slice = plot_slice(study)
            fig_slice.write_html("slice_plot.html")
            mlflow.log_artifact("slice_plot.html")
            os.remove("slice_plot.html")

            if len(study.trials) > 1:
                fig_imp = plot_param_importances(study)
                fig_imp.write_html("param_importances.html")
                mlflow.log_artifact("param_importances.html")
                os.remove("param_importances.html")
        except Exception as e:
            print(f"Could not generate or log Optuna plots: {e}")

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_rmse_cv_mean", study.best_value)
        mlflow.log_metric(
            "best_rmse_cv_var", study.best_trial.user_attrs["rmse_cv_var"]
        )
        mlflow.log_metric(
            "best_rmse_final_model", study.best_trial.user_attrs["rmse_final_model"]
        )

        # Retrain best model on full training set and log as artifact
        best_param_set = {**base_param_set, **study.best_params}
        data, _ = create_feature_subset(best_param_set, training_data)

        # For final model, we usually train on everything before the final evaluation split if we are just logging the "best" model from tuning
        # but to be consistent with train_flow, we can train on the whole train_df (before 2025-01-01)
        split_ts = pd.Timestamp("2025-01-01", tz="CET")
        train_df = data.loc[data.index < split_ts]
        test_df = data.loc[data.index >= split_ts]

        training_config = _model_train_mapping[model_type](
            model_type=model_type,
            param_set=best_param_set,
            training_data=train_df,
            testing_data=test_df,
        )
        best_model, _ = training_config.fit_and_evaluate_final_model()

        model_path = "ebm_dummy.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        mlflow.log_artifact(model_path)
        os.remove(model_path)

        print(f"Best trial: {study.best_trial.number}")
        print(f"  Value: {study.best_value}")
        print(f"  Params: {study.best_params}")

    return study


if __name__ == "__main__":
    from selected_feature_sets import catboost_restricted

    training_data = pd.read_pickle("cached_training_data.pkl")
    grid = build_config_from_features(catboost_restricted, target="prod_act_sum")
    run_bayesian_tuning(
        model_type="CatBoost",
        feature_set=grid,
        feature_set_name="catboost_restricted",
        training_data=training_data,
        n_trials=25,
        experiment_name="bayesian_tuning",
        create_new_study=True,
    )
