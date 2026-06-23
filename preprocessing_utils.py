from typing import List, Optional

import numpy as np
import pandas as pd
import holidays
from pandas import DataFrame


def create_time_features(
    df: pd.DataFrame, feature_types: List[str], datetime_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Creates cyclic time features from a DataFrame with a DateTime index or column.

    Cyclic encoding uses sine and cosine to capture the periodic nature of
    time features (e.g., day 1 and day 31 of a month are close to each other).

    Args:
        df: Input DataFrame with a DateTime index or column.
        feature_types: List of features to create. Possible values:
            - 'day_of_week': Day of the week (0-6)
            - 'month_of_year': Month (1-12)
            - 'hour_of_day': Hour (0-23)
            - 'day_of_month': Day of the month (1-31)
            - 'day_of_year': Day of the year (1-366)
            - 'hour_of_year': Hour of the year (0-8783)
            - 'week_of_year': Week of the year (1-53)
            - 'quarter': Quarter (1-4)
        datetime_column: Name of the DateTime column (if index is not used).

    Returns:
        DataFrame with original data and new cyclic features.

    Example:
        >>> df = pd.DataFrame({'value': [1, 2, 3]},
        ...                   index=pd.date_range('2024-01-01', periods=3, freq='D'))
        >>> result = create_time_features(df, ['day_of_week', 'month_of_year'])
        >>> print(result.columns)
        Index(['value', 'day_of_week_sin', 'day_of_week_cos',
               'month_of_year_sin', 'month_of_year_cos'], dtype='object')
    """
    df_result = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "DataFrame index must be a DatetimeIndex or datetime_column must be specified"
        )
    dt_obj = df.index

    feature_config = {
        "day_of_week": (7, lambda x: x.dayofweek),
        "month_of_year": (12, lambda x: x.month),
        "hour_of_day": (24, lambda x: x.hour),
        "day_of_month": (31, lambda x: x.day),
        "day_of_year": (366, lambda x: x.dayofyear),
        "hour_of_year": (366 * 24, lambda x: (x.dayofyear - 1) * 24 + x.hour),
        "week_of_year": (
            53,
            lambda x: x.isocalendar().week if hasattr(x, "isocalendar") else x.week,
        ),
        "quarter": (4, lambda x: x.quarter),
    }

    de_holidays = holidays.Germany()

    for feature_type in feature_types:
        if feature_type == "weekend_or_holiday":
            is_weekend = dt_obj.dayofweek >= 5
            is_holiday = pd.Series(dt_obj).apply(lambda x: x in de_holidays).values
            df_result["weekend_or_holiday"] = (is_weekend | is_holiday).astype(int)
            continue

        if feature_type not in feature_config:
            raise ValueError(
                f"Unknown feature type: {feature_type}. "
                f"Possible values: {list(feature_config.keys())}"
            )

        max_val, extract_func = feature_config[feature_type]

        values = extract_func(dt_obj)

        if feature_type in ["month_of_year", "day_of_month", "quarter"]:
            values = values - 1

        angle = 2 * np.pi * values / max_val

        df_result[f"{feature_type}_sin"] = np.sin(angle)
        df_result[f"{feature_type}_cos"] = np.cos(angle)

    return df_result


def create_feature_subset(param_set, training_data: DataFrame) -> tuple[DataFrame, str]:
    cyclic_time_features = param_set.get("cyclic_time_features")
    if cyclic_time_features:
        training_data = create_time_features(training_data, cyclic_time_features)

    target_col = param_set["target"]

    time_feature_cols = []
    if cyclic_time_features:
        for ft in cyclic_time_features:
            if ft == "weekend_or_holiday":
                time_feature_cols.append(ft)
            else:
                time_feature_cols.extend([f"{ft}_sin", f"{ft}_cos"])

    requested_features = set()
    for cov_list in [
        param_set.get("past_covariates", []),
        param_set.get("future_covariates", []),
    ]:
        for cov_dict in cov_list:
            for base, config in cov_dict.items():
                for lag in config.get("lags", []):
                    requested_features.add(f"{base}_lag_{lag}_hours")

                windows_config = config.get("windows", {})
                for win_type, win_val in windows_config.items():
                    for size in win_val.get("sizes", []):
                        size_str = (
                            f"{size[0]}_{size[1]}"
                            if isinstance(size, (list, tuple))
                            else str(size)
                        )
                        for agg in win_val.get("aggs", []):
                            requested_features.add(
                                f"{base}_w_{win_type}_{size_str}_{agg}"
                            )

    missing = requested_features - set(training_data.columns)
    if missing:
        raise ValueError(
            f"Requested features not found in training data: {sorted(missing)}"
        )

    feature_cols = []
    for col in training_data.columns:
        if col == target_col or col in time_feature_cols:
            continue

        if col in requested_features:
            feature_cols.append(col)

    data = training_data[[target_col] + feature_cols + time_feature_cols]
    return data, target_col
