import json
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tempfile
from typing import Any, List
from mlflow.entities import Experiment
from pandas import DataFrame
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from catboost import Pool, EFeaturesSelectionAlgorithm, EShapCalcType, CatBoostRegressor

from preprocessing_utils import create_feature_subset
from training import (
    SKLearnLikeModelTrainingConfig,
    SelfExplainMLModelTrainingConfig,
    _model_mapping,
    _model_train_mapping,
)

LAG_GENERATORS = {
    "hour_of_day": lambda size: (
        list(range(size)) if isinstance(size, int) else list(range(size[0], size[1]))
    ),
    "hour_of_weekday": lambda size: (
        list(range(size * 24)) if isinstance(size, int) else list(range(size[0] * 24, size[1] * 24))
    ),
    "past_hours": lambda size: (
        list(range(size[0], size[1])) if isinstance(size, (list, tuple)) else list(range(size))
    ),
}


def get_max_lag_from_grid(grid: dict) -> int:
    max_lag = 0

    def process_cov_dict(cov_dict):
        nonlocal max_lag
        for cov_cfg in cov_dict.values():
            if "lags" in cov_cfg:
                if cov_cfg["lags"]:
                    max_lag = max(max_lag, max(cov_cfg["lags"]))
            if "windows" in cov_cfg:
                for win_type, win_cfg in cov_cfg["windows"].items():
                    if win_type in LAG_GENERATORS:
                        for size in win_cfg.get("sizes", []):
                            lags = LAG_GENERATORS[win_type](size)
                            if lags:
                                max_lag = max(max_lag, max(lags))

    if "past_covariates" in grid:
        for sublist in grid["past_covariates"]:
            for cov_dict in sublist:
                process_cov_dict(cov_dict)

    if "future_covariates" in grid:
        for sublist in grid["future_covariates"]:
            for cov_dict in sublist:
                process_cov_dict(cov_dict)

    return max_lag


def get_kitchen_sink_grid(target: str = "prod_act_sum") -> dict:
    """
    Returns the 'Kitchen Sink' feature grid configuration.
    """
    return {
        "target": [target],
        "past_covariates": [
            [
                {
                    "prod_act_sum": {
                        "lags": [72, 96, 168],
                        "windows": {
                            "hour_of_day": {
                                "sizes": [
                                    [72, 168],
                                    [72, 168 + 7 * 24],
                                    [72, 168 + 14 * 24],
                                    [72, 168 + 28 * 24],
                                ],
                                "aggs": ["mean", "std", "slope"],
                            },
                            "hour_of_weekday": {
                                "sizes": [4, 8, 12],
                                "aggs": ["mean", "std", "slope"],
                            },
                        },
                    },
                    "missing_pinst_sum": {
                        "lags": list(range(42, 73)) + [168],
                        "windows": {
                            "hour_of_day": {
                                "sizes": [7, 14, 28],
                                "aggs": ["mean", "std", "slope"],
                            },
                            "hour_of_weekday": {
                                "sizes": [4, 8],
                                "aggs": ["mean", "std", "slope"],
                            },
                            "past_hours": {
                                "sizes": [[42, 48], [42, 48, 168]],
                                "aggs": ["mean", "std", "slope"],
                            },
                        },
                    },
                    "spot_price_act": {
                        "lags": list(range(42, 49)) + [168],
                        "windows": {
                            "hour_of_day": {
                                "sizes": [7, 14, 28],
                                "aggs": ["mean", "std", "slope"],
                            },
                            "past_hours": {
                                "sizes": [[42, 48], [42, 48, 168]],
                                "aggs": ["mean", "std", "slope"],
                            },
                        },
                    },
                    "rebap_price_est": {
                        "lags": list(range(42, 49)) + [168],
                        "windows": {
                            "hour_of_day": {
                                "sizes": [7, 14, 28],
                                "aggs": ["mean", "std", "slope"],
                            },
                            "past_hours": {
                                "sizes": [[42, 48], [42, 48, 168]],
                                "aggs": ["mean", "std", "slope"],
                            },
                        },
                    },
                    "fcr_price_act": {
                        "lags": list(range(42, 49)) + [168],
                        "windows": {
                            "hour_of_day": {
                                "sizes": [7, 14, 28],
                                "aggs": ["mean", "std", "slope"],
                            },
                            "past_hours": {
                                "sizes": [[42, 48], [42, 48, 168]],
                                "aggs": ["mean", "std", "slope"],
                            },
                        },
                    },
                    "afrr_neg_avg_price_act": {
                        "lags": list(range(42, 49)) + [168],
                        "windows": {
                            "hour_of_day": {
                                "sizes": [7, 14, 28],
                                "aggs": ["mean", "std", "slope"],
                            },
                            "past_hours": {
                                "sizes": [[42, 48], [42, 48, 168]],
                                "aggs": ["mean", "std", "slope"],
                            },
                        },
                    },
                    "afrr_pos_avg_price_act": {
                        "lags": list(range(42, 49)) + [168],
                        "windows": {
                            "hour_of_day": {
                                "sizes": [7, 14, 28],
                                "aggs": ["mean", "std", "slope"],
                            },
                            "past_hours": {
                                "sizes": [[42, 48], [42, 48, 168]],
                                "aggs": ["mean", "std", "slope"],
                            },
                        },
                    },
                }
            ]
        ],
        "future_covariates": [
            [
                {
                    "pinst_active_sum": {"lags": [0]},
                    "log_wind_fc": {
                        "lags": [0, 1, 2, 24, 168],
                        "windows": {
                            "hour_of_day": {
                                "sizes": [7, 14],
                                "aggs": ["mean", "std", "slope"],
                            },
                            "past_hours": {
                                "sizes": [[0, 24], [0, 48]],
                                "aggs": ["mean", "std", "slope"],
                            },
                        },
                    },
                    "prod_fc_sum": {
                        "lags": [0, 1, 2, 24, 48, 72, 168],
                        "windows": {
                            "hour_of_day": {
                                "sizes": [7, 14],
                                "aggs": ["mean", "std", "slope"],
                            },
                            "past_hours": {
                                "sizes": [[0, 24], [0, 48]],
                                "aggs": ["mean", "std", "slope"],
                            },
                        },
                    },
                    "temperature_fc": {
                        "lags": [0, 1, 2],
                        "windows": {
                            "hour_of_day": {
                                "sizes": [7, 14],
                                "aggs": ["mean", "std", "slope"],
                            },
                            "past_hours": {
                                "sizes": [[0, 24], [0, 48]],
                                "aggs": ["mean", "std", "slope"],
                            },
                        },
                    },
                    "solar_fc": {
                        "lags": [0, 1, 2, 24, 168],
                        "windows": {
                            "hour_of_day": {
                                "sizes": [7, 14],
                                "aggs": ["mean", "std", "slope"],
                            },
                            "past_hours": {
                                "sizes": [[0, 24], [0, 48]],
                                "aggs": ["mean", "std", "slope"],
                            },
                        },
                    },
                    "rebap_price_est": {"lags": [0]},
                }
            ]
        ],
        "cyclic_time_features": [
            ["day_of_week", "hour_of_day", "day_of_year", "weekend_or_holiday"]
        ],
        "scaler": [["minmax"]],
    }


def build_config_from_features(feature_list: List[str], target: str) -> dict:
    """
    Reconstructs a grid dictionary from a list of feature names.
    """
    past_covariates = {}
    future_covariates = {}
    cyclic_time_features = set()

    future_bases = ["prod_fc_sum", "temperature_fc", "solar_fc", "log_wind_fc"]
    past_bases = [
        "prod_act_sum",
        "rebap_price_est",
        "fcr_price_act",
        "afrr_neg_avg_price_act",
        "afrr_pos_avg_price_act",
        "missing_pinst_sum",
        "pinst_active_sum",
        "spot_price_act",
    ]

    for feature in feature_list:
        if any(
            feature.startswith(c)
            for c in [
                "day_of_week",
                "hour_of_day",
                "day_of_year",
                "weekend_or_holiday",
            ]
        ):
            for c in [
                "day_of_week",
                "hour_of_day",
                "day_of_year",
                "weekend_or_holiday",
            ]:
                if feature.startswith(c):
                    cyclic_time_features.add(c)
            continue

        base = None
        for b in future_bases + past_bases:
            if feature.startswith(b):
                base = b
                break

        if not base:
            continue

        target_dict = future_covariates if base in future_bases else past_covariates
        if base not in target_dict:
            target_dict[base] = {"lags": [], "windows": {}}

        if "_lag_" in feature:
            try:
                lag = int(feature.split("_lag_")[1].split("_")[0])
                if lag not in target_dict[base]["lags"]:
                    target_dict[base]["lags"].append(lag)
            except (ValueError, IndexError):
                pass
        elif "_w_" in feature:
            # Feature shape: f"{base}_w_{win_type}_{size_str}_{agg}"
            #   win_type ∈ LAG_GENERATORS keys (multi-token, e.g. "hour_of_day")
            #   size_str is "<int>" or "<int>_<int>"  (range windows like [72, 168])
            #   agg is a single token, e.g. "mean", "std", "slope"
            # Naive split-on-"_" misclassifies the leading number of a range size as
            # part of the win_type — match win_types by exact prefix instead.
            known_win_types = sorted(LAG_GENERATORS.keys(), key=len, reverse=True)
            known_aggs = ("mean", "std", "slope")

            middle = feature[len(f"{base}_w_") :]
            agg = next((a for a in known_aggs if middle.endswith(f"_{a}")), None)
            if agg is None:
                continue
            without_agg = middle[: -(len(agg) + 1)]
            win_type = next(
                (w for w in known_win_types if without_agg.startswith(f"{w}_")),
                None,
            )
            if win_type is None:
                continue
            size_str = without_agg[len(win_type) + 1 :]

            if "_" in size_str:
                try:
                    size = [int(x) for x in size_str.split("_")]
                except ValueError:
                    size = size_str
            else:
                try:
                    size = int(size_str)
                except ValueError:
                    size = size_str

            if win_type not in target_dict[base]["windows"]:
                target_dict[base]["windows"][win_type] = {"sizes": [], "aggs": []}

            if size not in target_dict[base]["windows"][win_type]["sizes"]:
                target_dict[base]["windows"][win_type]["sizes"].append(size)
            if agg not in target_dict[base]["windows"][win_type]["aggs"]:
                target_dict[base]["windows"][win_type]["aggs"].append(agg)

    grid = {
        "target": [target],
        "cyclic_time_features": [list(cyclic_time_features)],
        "scaler": [["minmax"]],
    }
    if past_covariates:
        grid["past_covariates"] = [[past_covariates]]
    if future_covariates:
        grid["future_covariates"] = [[future_covariates]]
    return grid


def find_elbow(ks, scores):
    ks = np.array(ks)
    scores = np.array(scores)
    if len(ks) < 3:
        return ks[-1]
    ks_norm = (ks - ks.min()) / (ks.max() - ks.min())
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
    coords = np.vstack((ks_norm, scores_norm)).T
    first_point = coords[0]
    line_vec = coords[-1] - coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = coords - first_point
    scalar_proj = np.dot(vec_from_first, line_vec_norm)
    vec_proj = np.outer(scalar_proj, line_vec_norm)
    vec_to_line = vec_from_first - vec_proj
    dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))
    return ks[np.argmax(dist_to_line)]


def run_feature_selection(
    model_type_sfs: str = "CatBoost",
    target: str = "prod_act_sum",
    experiment_name: str = "feature_selection_funnel",
    use_cached_data: bool = True,
    cache_loaded_data: bool = False,
):
    grid = get_kitchen_sink_grid(target)
    base_start = pd.Timestamp(year=2022, month=1, day=1, tz="CET")
    max_lag = get_max_lag_from_grid(grid)
    start = max(
        base_start + pd.Timedelta(hours=max_lag),
        pd.Timestamp(year=2022, month=2, day=1, tz="CET"),
    )
    end = pd.Timestamp(year=2024, month=12, day=31, tz="CET")

    mlflow.set_tracking_uri(uri="http://localhost:5000")
    if not (mlflow.get_experiment_by_name(experiment_name)):
        mlflow.create_experiment(experiment_name)
    exp = mlflow.set_experiment(experiment_name=experiment_name)

    base_path = os.path.dirname(os.path.abspath(__file__))
    cache_file = os.path.join(base_path, "cached_feature_selection_data.pkl")

    if use_cached_data:
        training_data = pd.read_pickle(cache_file)
    else:
        raise NotImplementedError("Data loading from internal systems removed. Use cached_feature_selection_data.pkl.")

    training_data = training_data.loc[training_data.index >= start]
    training_data = training_data.loc[training_data.index <= end]

    param_set = list(ParameterGrid(grid))[0]

    with mlflow.start_run(experiment_id=exp.experiment_id):
        # Stage 1: CatBoost Pruning
        print("Stage 1: CatBoost importance filtering...")
        data, target_col = create_feature_subset(param_set, training_data)
        training_config = SKLearnLikeModelTrainingConfig(
            model_type="CatBoost",
            param_set=param_set,
            training_data=data,
        )

        # Split for select_features (RFE)
        split_idx = int(len(data) * 0.8)
        train_df = training_config.training_data.iloc[:split_idx]
        val_df = training_config.training_data.iloc[split_idx:]
        train_scaled, val_scaled, target_scaler, covariate_scaler = (
            training_config.scale_data(train_df, val_df)
        )

        covariates = training_config.covariates
        X_train = train_scaled[covariates]
        y_train = train_scaled[training_config.target_col]
        X_val = val_scaled[covariates]
        y_val = val_scaled[training_config.target_col]

        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)

        model = training_config.create_model()
        summary = model.select_features(
            train_pool,
            eval_set=val_pool,
            features_for_select=list(range(len(covariates))),
            num_features_to_select=min(20, len(covariates)),
            steps=3,
            algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
            shap_calc_type=EShapCalcType.Regular,
            train_final_model=False,
            logging_level="Silent",
        )

        # Also get importance > 0.1
        full_model = training_config.fit_final_model()
        importances = full_model.get_feature_importance()
        importance_df = pd.DataFrame({"feature": covariates, "importance": importances})
        high_importance_features = importance_df[importance_df["importance"] > 0.1][
            "feature"
        ].tolist()

        # Combine high importance and RFE selected features
        rfe_selected = [covariates[i] for i in summary["selected_features"]]
        initial_features = list(set(high_importance_features) | set(rfe_selected))
        print(
            f"Stage 1 complete. Funnelled {len(covariates)} features down to {len(initial_features)}."
        )

        # Stage 2: SFS
        print(f"Stage 2: Sequential Feature Selection with {model_type_sfs}...")
        reduced_grid = build_config_from_features(initial_features, target)
        reduced_param_set = list(ParameterGrid(reduced_grid))[0]

        # Inject SFS specific hyperparams into param_set
        if model_type_sfs in ["CatBoost", "IGANN"]:
            reduced_param_set["n_estimators"] = 200

        reduced_data, _ = create_feature_subset(reduced_param_set, training_data)

        # Use training config for imputation and model creation
        sfs_training_config = _model_train_mapping[model_type_sfs](
            model_type=model_type_sfs,
            param_set=reduced_param_set,
            training_data=reduced_data,
        )
        cleaned_reduced_data = sfs_training_config.training_data

        X = cleaned_reduced_data.drop(columns=[target_col])
        y = cleaned_reduced_data[target_col].values.ravel()

        # Setup SFS Model using training config
        if isinstance(sfs_training_config, SelfExplainMLModelTrainingConfig):
            # Self-explain models need meta_info and benefit from scaling
            scaled_data, _, scaler_y, scaler_x = sfs_training_config.scale_data(
                sfs_training_config.training_data
            )
            X = scaled_data.drop(columns=[target_col])
            y = scaled_data[target_col].values.ravel()
            meta_info = sfs_training_config.create_meta_info(scaler_x, scaler_y)
            sfs_estimator = sfs_training_config.create_model(meta_info=meta_info)
        else:
            sfs_estimator = sfs_training_config.create_model()

        tscv = TimeSeriesSplit(n_splits=5)

        sfs = SFS(
            estimator=sfs_estimator,
            k_features=(5, min(25, len(X.columns))),
            forward=True,
            floating=True,
            scoring="neg_root_mean_squared_error",
            cv=tscv,
            n_jobs=-1,
        )

        sfs.fit(X, y)
        print("SFS complete.")

        # Metrics and Logging
        metric_dict = sfs.get_metric_dict()
        num_features = sorted(metric_dict.keys())
        rmses = [-metric_dict[k]["avg_score"] for k in num_features]

        for k, rmse in zip(num_features, rmses):
            mlflow.log_metric("sfs_rmse", rmse, step=k)

        recommended_k = find_elbow(num_features, rmses)
        best_features = list(sfs.subsets_[recommended_k]["feature_names"])

        mlflow.log_params(
            {
                "model_type_sfs": model_type_sfs,
                "stage1_n_features": len(initial_features),
                "recommended_k": recommended_k,
                "sfs_scoring": "neg_root_mean_squared_error",
            }
        )

        # Save and log artifacts (no local persistence)
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Ranking
            ranking = []
            for k in num_features:
                ranking.append(
                    {
                        "n_features": k,
                        "rmse": -metric_dict[k]["avg_score"],
                        "features": list(sfs.subsets_[k]["feature_names"]),
                    }
                )
            ranking_path = os.path.join(tmp_dir, "sfs_feature_ranking.json")
            with open(ranking_path, "w") as f:
                json.dump(ranking, f, indent=4)
            mlflow.log_artifact(ranking_path)

            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(num_features, rmses, marker="o")
            plt.axvline(
                x=recommended_k,
                color="r",
                linestyle="--",
                label=f"Elbow ({recommended_k})",
            )
            plt.xlabel("Number of Features")
            plt.ylabel("RMSE")
            plt.title(f"SFS with {model_type_sfs} (RMSE)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(tmp_dir, "sfs_rmse_plot.png")
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path)

            # Recommended set
            rec_path = os.path.join(tmp_dir, "sfs_recommended_features.json")
            with open(rec_path, "w") as f:
                json.dump({"recommended_features": best_features}, f, indent=4)
            mlflow.log_artifact(rec_path)

        print(f"Feature selection complete. Recommended features: {best_features}")
        return best_features


if __name__ == "__main__":
    run_feature_selection(model_type_sfs="CatBoost", use_cached_data=True)
