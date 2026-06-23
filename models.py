from typing import Any, Literal
import tensorflow as tf
from catboost import CatBoostRegressor
from igann import IGANNRegressor as _IGANNRegressor


import pandas as pd


class IGANNRegressor(_IGANNRegressor):
    def __init__(
        self,
        task="regression",
        n_hid=10,
        n_estimators=5000,
        boost_rate=0.1,
        init_reg=1,
        elm_scale=1,
        elm_alpha=1,
        act="elu",
        early_stopping=50,
        device="cpu",
        random_state=42,
        verbose=0,
    ):
        super().__init__(
            task=task,
            n_hid=n_hid,
            n_estimators=n_estimators,
            boost_rate=boost_rate,
            init_reg=init_reg,
            elm_scale=elm_scale,
            elm_alpha=elm_alpha,
            act=act,
            early_stopping=early_stopping,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )
        self.task = task
        self.n_hid = n_hid
        self.n_estimators = n_estimators
        self.boost_rate = boost_rate
        self.init_reg = init_reg
        self.elm_scale = elm_scale
        self.elm_alpha = elm_alpha
        self.act = act
        self.early_stopping = early_stopping
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
        return super().fit(X, y)

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
        return super().predict(X)


from interpret.glassbox import ExplainableBoostingRegressor
from nam.wrapper import NAMRegressor
from pydantic import BaseModel, field_validator
from pygam import LinearGAM as _LinearGAM


class LinearGAM(_LinearGAM):
    def __init__(
        self,
        terms="auto",
        max_iter=100,
        tol=0.0001,
        lam=0.6,
        n_splines=20,
        spline_order=3,
        verbose=False,
        callbacks=["deviance", "diffs"],
        fit_intercept=True,
        scale=None,
        **kwargs,
    ):
        super().__init__(
            terms=terms,
            max_iter=max_iter,
            tol=tol,
            lam=lam,
            n_splines=n_splines,
            spline_order=spline_order,
            verbose=verbose,
            callbacks=callbacks,
            fit_intercept=fit_intercept,
            scale=scale,
            **kwargs,
        )
        self.terms = terms
        self.max_iter = max_iter
        self.tol = tol
        self.lam = lam
        self.n_splines = n_splines
        self.spline_order = spline_order
        self.verbose = verbose
        self.callbacks = callbacks
        self.fit_intercept = fit_intercept
        self.scale = scale


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from exnn import ExNN

from gaminet import GAMINet
from prophet import Prophet
import numpy as np


class ProphetRegressor:
    """sklearn-like adapter around prophet.Prophet so it slots into _model_mapping.

    Prophet expects a DataFrame with `ds` and `y`; covariates become regressors via
    `add_regressor` before fit. The DatetimeIndex is tz-stripped because Prophet's
    Stan backend rejects tz-aware timestamps.

    Implements the sklearn estimator API (`get_params`/`set_params`/
    `__sklearn_clone__`) so mlxtend.SFS, which clones the estimator per fold,
    can use it as well.
    """

    _PROPHET_HYPERPARAMS = (
        "growth",
        "changepoint_prior_scale",
        "seasonality_prior_scale",
        "holidays_prior_scale",
        "seasonality_mode",
        "daily_seasonality",
        "weekly_seasonality",
        "yearly_seasonality",
        "n_changepoints",
        "changepoint_range",
    )

    def __init__(
        self,
        growth: str = "linear",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        seasonality_mode: str = "additive",
        daily_seasonality: Any = "auto",
        weekly_seasonality: Any = "auto",
        yearly_seasonality: Any = "auto",
        n_changepoints: int = 25,
        changepoint_range: float = 0.8,
    ):
        self.growth = growth
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.seasonality_mode = seasonality_mode
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self._model: Prophet | None = None
        self._regressor_cols: list[str] = []

    def get_params(self, deep: bool = True) -> dict:
        return {p: getattr(self, p) for p in self._PROPHET_HYPERPARAMS}

    def set_params(self, **params) -> "ProphetRegressor":
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __sklearn_clone__(self) -> "ProphetRegressor":
        return ProphetRegressor(**self.get_params())

    @staticmethod
    def _ds_from_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return index.tz_localize(None) if index.tz is not None else index

    def fit(self, X: pd.DataFrame, y) -> "ProphetRegressor":
        if not isinstance(X, pd.DataFrame) or not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError(
                "ProphetRegressor.fit requires X to be a DataFrame with a DatetimeIndex."
            )
        self._regressor_cols = list(X.columns)
        df = pd.DataFrame({"ds": self._ds_from_index(X.index)})
        df["y"] = np.asarray(y).reshape(-1)
        for col in self._regressor_cols:
            df[col] = X[col].to_numpy()

        self._model = Prophet(**self.get_params())
        for col in self._regressor_cols:
            self._model.add_regressor(col)
        self._model.fit(df)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("ProphetRegressor.predict called before fit.")
        if not isinstance(X, pd.DataFrame) or not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError(
                "ProphetRegressor.predict requires X to be a DataFrame with a DatetimeIndex."
            )
        df = pd.DataFrame({"ds": self._ds_from_index(X.index)})
        for col in self._regressor_cols:
            df[col] = X[col].to_numpy()
        forecast = self._model.predict(df)
        return forecast["yhat"].to_numpy()


class EBMConfig(BaseModel):
    # most important hyperparams from https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html
    max_bins: int = 256
    interactions: int | str = "5x"
    outer_bags: int = 14
    inner_bags: int = 0
    learning_rate: float = 0.04
    early_stopping_rounds: int = 100
    max_leaves: int = 4
    random_state: int = 42


class LinearGAMConfig(BaseModel):
    terms: Any = "auto"
    max_iter: int = 100
    tol: float = 0.0001
    lam: float = 0.6
    n_splines: int = 20
    spline_order: int = 3


class LRConfig(BaseModel):
    l1_ratio: float = 0.5
    alpha: float = 1


class DTConfig(BaseModel):
    max_depth: int | None = None
    max_leaf_nodes: int | None = None
    splitter: Literal["best", "random"] = "best"
    random_state: int = 42


class RFConfig(BaseModel):
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: float | Literal["sqrt", "log2"] | None = 1.0
    random_state: int = 42


class MLPConfig(BaseModel):
    hidden_layer_sizes: tuple = (100,)
    activation: Literal["identity", "logistic", "tanh", "relu"] = "relu"
    alpha: float = 0.0001
    learning_rate_init: float = 0.001
    random_state: int = 42


class SVMConfig(BaseModel):
    kernel: Literal["linear", "poly", "rbf", "sigmoid"] = "rbf"
    C: float = 1.0
    epsilon: float = 0.1
    gamma: Literal["scale", "auto"] | float = "scale"


class XGBConfig(BaseModel):
    n_estimators: int | None = 100
    max_depth: int = 6
    learning_rate: float = 0.3
    random_state: int = 42


class GAMINetConfig(BaseModel):
    interact_num: int = 20
    reg_clarity: float = 0.1
    subnet_arch: list = [40] * 5
    interact_arch: list = [40] * 5
    lr_bp: list = [1e-4, 1e-4, 1e-4]
    batch_size: int = 200
    main_effect_epochs: int = 5000
    interaction_epochs: int = 5000
    tuning_epochs: int = 500
    early_stop_thres: list = [50, 50, 50]
    heredity: bool = True
    activation_func: Any = tf.nn.relu
    random_state: int = 42

    @field_validator("lr_bp", "early_stop_thres", mode="before")
    @classmethod
    def convert_to_list(cls, v):
        if isinstance(v, (float, int)):
            return [v, v, v]
        return v


class ExNNConfig(BaseModel):
    subnet_num: int = 5
    l1_proj: int = 0.001
    l1_subnet: float = 0.001
    random_state: int = 42


class CatBoostConfig(BaseModel):
    max_depth: int = 6
    eta: float = 0.03
    n_estimators: int = 1000
    random_seed: int = 42
    verbose: bool = False


class IGANNConfig(BaseModel):
    n_estimators: int = 5000
    boost_rate: float = 0.1
    init_reg: int = 1
    elm_scale: float = 1
    elm_alpha: float = 1
    act: str = "elu"
    early_stopping: int = 50
    random_state: int = 42


class ProphetConfig(BaseModel):
    growth: Literal["linear", "flat"] = "linear"
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    holidays_prior_scale: float = 10.0
    seasonality_mode: Literal["additive", "multiplicative"] = "additive"
    daily_seasonality: Any = "auto"
    weekly_seasonality: Any = "auto"
    yearly_seasonality: Any = "auto"
    n_changepoints: int = 25
    changepoint_range: float = 0.8


class NAMConfig(BaseModel):
    num_learners: int = 20
    dropout: float = 0.1
    num_basis_functions: int = 64
    lr: float = 0.02082
    metric: str = "rmse"
    random_state: int = 42


_model_mapping = {
    "LinearGAM": {"model": LinearGAM, "config": LinearGAMConfig},
    "SVM": {"model": SVR, "config": SVMConfig},
    "EBM": {"model": ExplainableBoostingRegressor, "config": EBMConfig},
    "ElasticNet": {"model": ElasticNet, "config": LRConfig},
    "DecisionTree": {"model": DecisionTreeRegressor, "config": DTConfig},
    "MLP": {"model": MLPRegressor, "config": MLPConfig},
    "RandomForest": {"model": RandomForestRegressor, "config": RFConfig},
    "XGBoost": {"model": XGBRegressor, "config": XGBConfig},
    "ExNN": {"model": ExNN, "config": ExNNConfig},
    "GAMINet": {"model": GAMINet, "config": GAMINetConfig},
    "CatBoost": {"model": CatBoostRegressor, "config": CatBoostConfig},
    "IGANN": {"model": IGANNRegressor, "config": IGANNConfig},
    "Prophet": {"model": ProphetRegressor, "config": ProphetConfig},
    # "NAM": {"model": NAMRegressor, "config": NAMConfig},
}
