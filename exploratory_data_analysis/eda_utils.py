import pandas as pd
import numpy as np
from pathlib import Path
from preprocessing_utils import create_time_features


def load_eda_data(start_date, end_date):
    df = pd.read_pickle(Path(__file__).parent / "df.pkl")
    df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    return df


def handle_outliers_backward_imputation(df, column, outlier_mask):
    """
    The affected values were replaced by backward imputation: Values from the same seasonal
    period of the previous year were identified and included in the correction with a weight of
    60%. In addition, the original value was included with a weight of 40% in order not to
    completely level out the actual trend.
    """
    corrected_series = df[column].copy()

    # We need to find the same seasonal period of the previous year.
    # For hourly data, that's approximately 365*24 hours ago,
    # but to be precise with seasonality (same hour, same day of year),
    # we can use shift or manual lookup.

    # Assuming the index is a DatetimeIndex with 1h frequency.
    # 1 year ago = 365 days * 24 hours = 8760 hours (not accounting for leap years)
    # Better: use pandas DateOffset

    for idx in df[outlier_mask].index:
        prev_year_idx = idx - pd.DateOffset(years=1)

        if prev_year_idx in df.index:
            prev_year_val = df.loc[prev_year_idx, column]
            original_val = df.loc[idx, column]

            # Correction: 60% prev_year_val + 40% original_val
            corrected_val = 0.6 * prev_year_val + 0.4 * original_val
            corrected_series.loc[idx] = corrected_val

    return corrected_series


if __name__ == "__main__":
    # Quick test
    start = pd.Timestamp("2022-01-01", tz="CET")
    end = pd.Timestamp("2023-12-31", tz="CET")
    # df = load_eda_data(start, end)
    # print(df.head())
    pass
