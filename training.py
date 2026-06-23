import gc
import pickle
from typing import Any, Tuple

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from mlflow.entities import Experiment
from mlflow.models import infer_signature
from pandas import DataFrame, Index
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from models import _model_mapping
from preprocessing_utils import create_feature_subset


def _log_model_artifact(final_model: Any, model_type: str) -> None:
    """GAMINet pickled via plain `pickle.dump` goes through keras's bytecode
    serializer, which writes an empty `get_config()` and drops every attribute
    `global_explain` / `local_explain` need (`data_dict_density`, `active_indices`,
    `effect_names`, `nfeature_scaler_`, ...). Use GAMINet's own `save()` for those;
    Prophet's Stan backend doesn't pickle reliably across versions, so use its
    own JSON serializer; everything else stays on plain pickle."""
    if model_type == "GAMINet":
        final_model.save(folder="./", name="model")  # writes ./model.pickle
        mlflow.log_artifact("model.pickle")
    elif model_type == "Prophet":
        from prophet.serialize import model_to_json

        with open("model.json", "w") as f:
            f.write(model_to_json(final_model._model))
        mlflow.log_artifact("model.json")
        with open("regressor_cols.json", "w") as f:
            import json

            json.dump(final_model._regressor_cols, f)
        mlflow.log_artifact("regressor_cols.json")
    else:
        with open("model.pkl", "wb") as f:
            pickle.dump(final_model, f)
        mlflow.log_artifact("model.pkl")


def train(
    exp: Experiment,
    model_type: str,
    param_set,
    training_data: DataFrame,
    n_splits=5,
    additional_tag: dict = None,
    only_final_model: bool = False,
):
    data, target_col = create_feature_subset(param_set, training_data)
    split_ts = pd.Timestamp("2025-01-01", tz="CET")
    train = data.loc[data.index < split_ts]
    test = data.loc[data.index >= split_ts]
    training_config: ModelTrainingConfig = _model_train_mapping[model_type](
        model_type=model_type,
        param_set=param_set,
        training_data=train,
        testing_data=test,
    )
    with mlflow.start_run(experiment_id=exp.experiment_id):
        if additional_tag:
            mlflow.set_tags(additional_tag)
        mlflow.log_params(training_config.data_params)
        mlflow.log_params(training_config.hyperparam_config.model_dump())
        mlflow.log_params({"scaler_x": training_config.scaler_type_x})
        mlflow.log_params({"scaler_y": training_config.scaler_type_y})

        mlflow.log_params(
            {
                "train_start": train.index[0],
                "train_end": train.index[-1],
                "test_start": test.index[0] if not test.empty else None,
                "test_end": test.index[-1] if not test.empty else None,
            }
        )

        if not only_final_model:
            rmse_mean, rmse_var = training_config.cross_validate(n_splits=n_splits)
        if (
            training_config.testing_data is not None
            and not training_config.testing_data.empty
        ):
            final_model, rmse = training_config.fit_and_evaluate_final_model()
            mlflow.log_metric("rmse_final_model", rmse)
        else:
            final_model = training_config.fit_final_model()
            rmse = None

        mlflow.set_tag("model_type", model_type)
        if len(training_config.hyper_params) == 0:
            mlflow.set_tag("default_parameters", True)

        if not only_final_model:
            mlflow.log_metric("rmse_cv_mean", rmse_mean)
            mlflow.log_metric("rmse_cv_var", rmse_var)

        _log_model_artifact(final_model, model_type)


class ModelTrainingConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_type: str
    param_set: dict[str, Any]
    training_data: DataFrame
    testing_data: DataFrame | None = None

    target_col: str = None
    covariates: Index = None
    scaler_type_x: str = None
    scaler_type_y: str = None
    data_params: dict[str, Any] = None
    hyper_params: dict[str, Any] = None
    hyperparam_config: BaseModel = None

    def __init__(
        self,
        model_type: str,
        param_set: dict[str, Any],
        training_data: DataFrame,
        testing_data: DataFrame | None = None,
    ):
        super().__init__(
            model_type=model_type,
            param_set=param_set,
            training_data=training_data,
            testing_data=testing_data,
        )
        self.clean_up_data()
        self.target_col = self.param_set["target"]
        self.covariates = training_data.drop(columns=[self.target_col]).columns

        self.scaler_type_x = self.param_set.get("scaler_x", ["minmax"])[
            0
        ]  # default to minmax
        self.scaler_type_y = self.param_set.get("scaler_y", ["minmax"])[
            0
        ]  # default to minmax

        # Extract hyperparameters: all params except target, past_covariates, future_covariates
        data_keys = {
            "target",
            "past_covariates",
            "future_covariates",
            "cyclic_time_features",
        }

        # Extract only target and covariates
        self.data_params = {k: v for k, v in self.param_set.items() if k in data_keys}
        self.hyper_params = {
            k: v for k, v in param_set.items() if k not in data_keys and k != "scaler"
        }

        self.hyperparam_config = _model_mapping[model_type]["config"](
            **self.hyper_params
        )

    def create_model(self) -> Any:
        raise NotImplementedError
    @staticmethod
    def _impute_prod_fc_sum_march_2023(data: DataFrame) -> None:
        """2023-03-02 to 2023-03-06: prod_fc_sum was ~194 MW below the prior-year
        forecast for every hour — a data pipeline anomaly, not real weather.
        Replace with values from 364 days (52 weeks) earlier to keep weekday alignment."""
        col = "prod_fc_sum_lag_0_hours"
        if col not in data.columns:
            return
        mask = (data.index >= pd.Timestamp("2023-03-02", tz="CET")) & (
            data.index <= pd.Timestamp("2023-03-06 23:00", tz="CET")
        )
        if not mask.any():
            return
        lag_364d = data[col].shift(364 * 24)
        n_imputed = mask.sum()
        data.loc[mask, col] = lag_364d.loc[mask]
        print(
            f"WARNING  : Imputed {n_imputed} values in {col} for 2023-03-02→06 "
            f"with same-weekday/hour values from 364 days prior."
        )

    @staticmethod
    def _impute_prod_fc_sum(data: DataFrame) -> None:
        col = "prod_fc_sum_lag_0_hours"
        if col not in data.columns:
            return
        suspect = data[col] < 100
        if not suspect.any():
            return
        lag_168h = data[col].shift(168)
        n_imputed = suspect.sum()
        data.loc[suspect, col] = lag_168h.loc[suspect]
        print(
            f"WARNING  : Imputed {n_imputed} suspect values in {col} (< 100) "
            f"with same-hour value from 168 h (1 week) prior."
        )

    def clean_up_data(self):
        # Python
        if self.testing_data is not None:
            if (
                self.training_data.isna().to_numpy().any()
                or self.testing_data.isna().to_numpy().any()
            ):
                nan_cols_train = self.training_data.columns[
                    self.training_data.isna().any()
                ].tolist()
                nan_cols_test = self.testing_data.columns[
                    self.testing_data.isna().any()
                ].tolist()
                nan_cols = list(set(nan_cols_train + nan_cols_test))

                df = pd.concat([self.training_data, self.testing_data], axis=0)

                # get the timestamps where any col is nan
                # We must only consider columns that are in nan_cols to avoid catching NaNs introduced by concat
                nan_rows = df[df[nan_cols].isna().any(axis=1)]
                # Filter the dict to only show the nan values for each row
                nan_rows_dict = {
                    ts.isoformat(): {
                        col: val for col, val in row.items() if pd.isna(val)
                    }
                    for ts, row in nan_rows[nan_cols].to_dict(orient="index").items()
                }
                print(
                    f"WARNING  : Training/Testing Data contains NaN-Values in the following columns: {nan_cols}."
                )
                print(f"NaN-Rows: {nan_rows_dict}")
        else:
            if self.training_data.isna().to_numpy().any():
                nan_cols = self.training_data.columns[
                    self.training_data.isna().any()
                ].tolist()
                nan_rows = self.training_data[
                    self.training_data[nan_cols].isna().any(axis=1)
                ]
                nan_rows_dict = {
                    ts.isoformat(): {
                        col: val for col, val in row.items() if pd.isna(val)
                    }
                    for ts, row in nan_rows[nan_cols].to_dict(orient="index").items()
                }
                print(
                    f"WARNING  : Training Data contains NaN-Values in the following columns: {nan_cols}."
                )
                print(f"NaN-Rows: {nan_rows_dict}")

        self._impute_prod_fc_sum_march_2023(self.training_data)
        self._impute_prod_fc_sum(self.training_data)
        if self.testing_data is not None:
            self._impute_prod_fc_sum_march_2023(self.testing_data)
            self._impute_prod_fc_sum(self.testing_data)


        # Filling NaNs with linear interpolation and then forward/backward fill for remaining NaNs at boundaries
        # This takes the mean of the values "around" the nan value
        self.training_data = self.training_data.interpolate(
            method="linear", limit_direction="both"
        )
        if self.testing_data is not None:
            self.testing_data = self.testing_data.interpolate(
                method="linear", limit_direction="both"
            )

    def cross_validate(self, n_splits: int = 5) -> tuple[float, float]:
        raise NotImplementedError

    def scale_data(
        self, training_data: DataFrame, testing_data: DataFrame | None = None
    ) -> tuple[
        DataFrame,
        DataFrame | None,
        MinMaxScaler | StandardScaler,
        MinMaxScaler | StandardScaler,
    ]:
        """
        Scales 2 sets of data. This function can be used for both cross-validation and final model training.

        Parameters:
        - training_data (DataFrame): Training dataset.
        - testing_data (DataFrame): Testing or validation dataset.

        Returns:
        - tuple[DataFrame, DataFrame | None, MinMaxScaler | StandardScaler]: Scaled training and testing/validation data along with the scaler used for features.
        """
        scaler_x = _scaler_map[self.scaler_type_x]()
        # sometimes self. covariates (from original data) is not the same as covariate columns in folds
        covariates = training_data.drop(columns=[self.target_col]).columns
        scaler_x.fit(training_data[covariates])
        scaler_y = _scaler_map[self.scaler_type_y]()
        scaler_y.fit(training_data[[self.target_col]])
        train_scaled = training_data.copy()

        train_scaled[covariates] = scaler_x.transform(training_data[covariates])
        train_scaled[[self.target_col]] = scaler_y.transform(
            training_data[[self.target_col]]
        )

        test_scaled = None
        if testing_data is not None:
            test_scaled = testing_data.copy()
            test_scaled[covariates] = scaler_x.transform(testing_data[covariates])
            test_scaled[[self.target_col]] = scaler_y.transform(
                testing_data[[self.target_col]]
            )

        return train_scaled, test_scaled, scaler_y, scaler_x

    def fit_and_evaluate_final_model(self) -> Any:
        raise NotImplementedError

    def fit_final_model(self) -> Any:
        raise NotImplementedError


class SKLearnLikeModelTrainingConfig(ModelTrainingConfig):
    def create_model(self) -> Any:
        return _model_mapping[self.model_type]["model"](
            **self.hyperparam_config.model_dump()
        )

    def cross_validate(self, n_splits: int = 5) -> tuple[float, float]:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        rmse_scores = []
        for train_index, valid_index in tscv.split(self.training_data):

            train_fold, valid_fold = (
                self.training_data.iloc[train_index],
                self.training_data.iloc[valid_index],
            )
            train_fold, valid_fold, target_scaler, covariate_scaler = self.scale_data(
                train_fold, valid_fold
            )
            model = self.create_model()
            model.fit(train_fold[self.covariates], train_fold[self.target_col])
            pred = model.predict(valid_fold[self.covariates])
            y_valid = np.array(valid_fold[self.target_col], dtype=np.float32).reshape(
                -1, 1
            )
            y_valid = target_scaler.inverse_transform(y_valid)

            pred = np.array(pred, dtype=np.float32).reshape(-1, 1)
            pred = target_scaler.inverse_transform(pred)
            rmse = root_mean_squared_error(y_valid, pred)
            rmse_scores.append(rmse)

        mean_rmse = float(np.mean(rmse_scores))
        variance_rmse = float(np.var(rmse_scores))
        return mean_rmse, variance_rmse

    def fit_and_evaluate_final_model(self) -> Tuple[Any, float]:

        train, test, target_scaler, covariate_scaler = self.scale_data(
            self.training_data, self.testing_data
        )
        model = self.create_model()
        model.fit(train[self.covariates], train[self.target_col])
        pred = model.predict(test[self.covariates])
        pred = np.array(pred, dtype=np.float32).reshape(-1, 1)
        pred = target_scaler.inverse_transform(pred)

        y_test = np.array(test[self.target_col], dtype=np.float32).reshape(-1, 1)
        y_test = target_scaler.inverse_transform(y_test)
        rmse = float(root_mean_squared_error(y_test, pred))
        return model, rmse

    def fit_final_model(self) -> Any:
        train, _, _, _ = self.scale_data(self.training_data)
        model = self.create_model()
        model.fit(train[self.covariates], train[self.target_col])
        return model


class SelfExplainMLModelTrainingConfig(ModelTrainingConfig):

    def create_model(self, meta_info: dict) -> Any:
        return _model_mapping[self.model_type]["model"](
            **self.hyperparam_config.model_dump(), meta_info=meta_info
        )

    def create_meta_info(
        self,
        scaler_x: MinMaxScaler | StandardScaler,
        scaler_y: MinMaxScaler | StandardScaler,
    ) -> dict:
        meta_info = {}

        # some self explain models expect a single column scaler for all features
        for i, col in enumerate(self.covariates):
            if pd.api.types.is_numeric_dtype(self.training_data[col]):
                meta_info[col] = {"type": "continuous"}
                if isinstance(scaler_x, MinMaxScaler):
                    feature_scaler = MinMaxScaler()
                    feature_scaler.min_ = np.array([scaler_x.min_[i]])
                    feature_scaler.scale_ = np.array([scaler_x.scale_[i]])
                    feature_scaler.data_min_ = np.array([scaler_x.data_min_[i]])
                    feature_scaler.data_max_ = np.array([scaler_x.data_max_[i]])
                    feature_scaler.data_range_ = np.array([scaler_x.data_range_[i]])
                    feature_scaler.n_samples_seen_ = scaler_x.n_samples_seen_
                    meta_info[col]["scaler"] = feature_scaler
                elif isinstance(scaler_x, StandardScaler):
                    feature_scaler = StandardScaler()
                    feature_scaler.mean_ = np.array([scaler_x.mean_[i]])
                    feature_scaler.var_ = np.array([scaler_x.var_[i]])
                    feature_scaler.scale_ = np.array([scaler_x.scale_[i]])
                    feature_scaler.n_samples_seen_ = scaler_x.n_samples_seen_
                    meta_info[col]["scaler"] = feature_scaler
                else:
                    meta_info[col]["scaler"] = scaler_x

            else:
                meta_info[col] = {"type": "categorical"}
                meta_info[col]["values"] = [1, 0]

        # This needs to be the last entry in meta_info or else it breaks something in the fit function of the SelfExplain-Models
        meta_info[self.target_col] = {"type": "target"}
        meta_info[self.target_col]["scaler"] = scaler_y

        return meta_info

    def cross_validate(self, n_splits: int = 5) -> tuple[float, float]:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        rmse_scores = []
        for train_index, valid_index in tscv.split(self.training_data):

            train_fold, valid_fold = (
                self.training_data.iloc[train_index],
                self.training_data.iloc[valid_index],
            )

            train_fold, valid_fold, scaler_y, scaler_x = self.scale_data(
                train_fold, valid_fold
            )
            meta_info = self.create_meta_info(scaler_x, scaler_y)
            model = _model_mapping[self.model_type]["model"](
                **self.hyperparam_config.model_dump(), meta_info=meta_info
            )
            train_fold_x = train_fold[self.covariates].to_numpy()
            train_fold_y = train_fold[self.target_col].to_numpy().reshape(-1, 1)
            valid_fold_x = valid_fold[self.covariates].to_numpy()
            valid_fold_y = valid_fold[self.target_col].to_numpy().reshape(-1, 1)

            model.fit(train_fold_x, train_fold_y)
            pred = model.predict(valid_fold_x)
            y_valid = np.array(valid_fold_y, dtype=np.float32).reshape(-1, 1)
            y_valid = scaler_y.inverse_transform(y_valid)

            pred = np.array(pred, dtype=np.float32).reshape(-1, 1)
            pred = scaler_y.inverse_transform(pred)
            rmse = root_mean_squared_error(y_valid, pred)
            rmse_scores.append(rmse)

            # GAMINet/ExNN are TF/Keras models — without explicit cleanup the
            # graph + optimizer state from each fold accumulates and OOMs the
            # process within a few Optuna trials.
            del (
                model,
                train_fold,
                valid_fold,
                train_fold_x,
                train_fold_y,
                valid_fold_x,
                valid_fold_y,
                pred,
                meta_info,
                scaler_x,
                scaler_y,
            )
            tf.keras.backend.clear_session()
            gc.collect()

        mean_rmse = float(np.mean(rmse_scores))
        variance_rmse = float(np.var(rmse_scores))
        return mean_rmse, variance_rmse

    def fit_and_evaluate_final_model(self) -> Tuple[Any, float]:
        if self.testing_data is None:
            raise ValueError(
                "testing_data must be provided to fit and evaluate final model"
            )
        train, test, target_scaler, covariate_scaler = self.scale_data(
            self.training_data, self.testing_data
        )
        meta_info = self.create_meta_info(covariate_scaler, target_scaler)
        model = self.create_model(meta_info=meta_info)

        train_x = train[self.covariates].to_numpy()
        train_y = train[self.target_col].to_numpy().reshape(-1, 1)
        test_x = test[self.covariates].to_numpy()
        test_y = test[self.target_col].to_numpy().reshape(-1, 1)
        model.fit(train_x, train_y)
        pred = model.predict(test_x)
        pred = np.array(pred, dtype=np.float32).reshape(-1, 1)
        pred = target_scaler.inverse_transform(pred)
        test_y = np.array(test_y, dtype=np.float32).reshape(-1, 1)
        test_y = target_scaler.inverse_transform(test_y)
        rmse = float(root_mean_squared_error(test_y, pred))
        return model, rmse

    def fit_final_model(self) -> Any:
        train, _, target_scaler, covariate_scaler = self.scale_data(self.training_data)
        meta_info = self.create_meta_info(covariate_scaler, target_scaler)
        model = self.create_model(meta_info=meta_info)

        train_x = train[self.covariates].to_numpy()
        train_y = train[self.target_col].to_numpy().reshape(-1, 1)
        model.fit(train_x, train_y)
        return model


_model_train_mapping = {
    "LinearGAM": SKLearnLikeModelTrainingConfig,
    "SVM": SKLearnLikeModelTrainingConfig,
    "EBM": SKLearnLikeModelTrainingConfig,
    "ElasticNet": SKLearnLikeModelTrainingConfig,
    "DecisionTree": SKLearnLikeModelTrainingConfig,
    "MLP": SKLearnLikeModelTrainingConfig,
    "RandomForest": SKLearnLikeModelTrainingConfig,
    "XGBoost": SKLearnLikeModelTrainingConfig,
    "ExNN": SelfExplainMLModelTrainingConfig,
    "GAMINet": SelfExplainMLModelTrainingConfig,
    "CatBoost": SKLearnLikeModelTrainingConfig,
    "IGANN": SKLearnLikeModelTrainingConfig,
    "Prophet": SKLearnLikeModelTrainingConfig,
    # "NAM": SKLearnLikeModelTrainingConfig,
}

_scaler_map = {"minmax": MinMaxScaler, "standard": StandardScaler}
