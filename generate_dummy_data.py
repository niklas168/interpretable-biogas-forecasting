"""Generate dummy data for intrepretml-public.

Produces:
- cached_training_data.pkl (34321 rows, 245 columns)
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# Exact column order from source project (245 columns)
CACHED_TRAINING_DATA_COLS = [
    'prod_act_sum', 'pinst_active_sum_lag_0_hours', 'log_wind_fc_lag_0_hours',
    'log_wind_fc_lag_1_hours', 'log_wind_fc_lag_2_hours', 'log_wind_fc_lag_24_hours',
    'log_wind_fc_lag_168_hours', 'log_wind_fc_w_hour_of_day_7_mean',
    'log_wind_fc_w_hour_of_day_7_std', 'log_wind_fc_w_hour_of_day_7_slope',
    'log_wind_fc_w_hour_of_day_14_mean', 'log_wind_fc_w_hour_of_day_14_std',
    'log_wind_fc_w_hour_of_day_14_slope', 'log_wind_fc_w_past_hours_0_24_mean',
    'log_wind_fc_w_past_hours_0_24_std', 'log_wind_fc_w_past_hours_0_24_slope',
    'log_wind_fc_w_past_hours_0_48_mean', 'log_wind_fc_w_past_hours_0_48_std',
    'log_wind_fc_w_past_hours_0_48_slope', 'prod_fc_sum_lag_0_hours',
    'prod_fc_sum_lag_1_hours', 'prod_fc_sum_lag_2_hours', 'prod_fc_sum_lag_24_hours',
    'prod_fc_sum_lag_48_hours', 'prod_fc_sum_lag_72_hours', 'prod_fc_sum_lag_168_hours',
    'prod_fc_sum_w_hour_of_day_7_mean', 'prod_fc_sum_w_hour_of_day_7_std',
    'prod_fc_sum_w_hour_of_day_7_slope', 'prod_fc_sum_w_hour_of_day_14_mean',
    'prod_fc_sum_w_hour_of_day_14_std', 'prod_fc_sum_w_hour_of_day_14_slope',
    'prod_fc_sum_w_past_hours_0_24_mean', 'prod_fc_sum_w_past_hours_0_24_std',
    'prod_fc_sum_w_past_hours_0_24_slope', 'prod_fc_sum_w_past_hours_0_48_mean',
    'prod_fc_sum_w_past_hours_0_48_std', 'prod_fc_sum_w_past_hours_0_48_slope',
    'temperature_fc_lag_0_hours', 'temperature_fc_lag_1_hours', 'temperature_fc_lag_2_hours',
    'temperature_fc_w_hour_of_day_7_mean', 'temperature_fc_w_hour_of_day_7_std',
    'temperature_fc_w_hour_of_day_7_slope', 'temperature_fc_w_hour_of_day_14_mean',
    'temperature_fc_w_hour_of_day_14_std', 'temperature_fc_w_hour_of_day_14_slope',
    'temperature_fc_w_past_hours_0_24_mean', 'temperature_fc_w_past_hours_0_24_std',
    'temperature_fc_w_past_hours_0_24_slope', 'temperature_fc_w_past_hours_0_48_mean',
    'temperature_fc_w_past_hours_0_48_std', 'temperature_fc_w_past_hours_0_48_slope',
    'solar_fc_lag_0_hours', 'solar_fc_lag_1_hours', 'solar_fc_lag_2_hours',
    'solar_fc_lag_24_hours', 'solar_fc_lag_168_hours', 'solar_fc_w_hour_of_day_7_mean',
    'solar_fc_w_hour_of_day_7_std', 'solar_fc_w_hour_of_day_7_slope',
    'solar_fc_w_hour_of_day_14_mean', 'solar_fc_w_hour_of_day_14_std',
    'solar_fc_w_hour_of_day_14_slope', 'solar_fc_w_past_hours_0_24_mean',
    'solar_fc_w_past_hours_0_24_std', 'solar_fc_w_past_hours_0_24_slope',
    'solar_fc_w_past_hours_0_48_mean', 'solar_fc_w_past_hours_0_48_std',
    'solar_fc_w_past_hours_0_48_slope', 'rebap_price_est_lag_0_hours', 'prod_act_sum_lag_72_hours',
    'prod_act_sum_lag_96_hours', 'prod_act_sum_lag_168_hours',
    'prod_act_sum_w_hour_of_day_72_168_mean', 'prod_act_sum_w_hour_of_day_72_168_std',
    'prod_act_sum_w_hour_of_day_72_168_slope', 'prod_act_sum_w_hour_of_day_72_336_mean',
    'prod_act_sum_w_hour_of_day_72_336_std', 'prod_act_sum_w_hour_of_day_72_336_slope',
    'prod_act_sum_w_hour_of_day_72_504_mean', 'prod_act_sum_w_hour_of_day_72_504_std',
    'prod_act_sum_w_hour_of_day_72_504_slope', 'prod_act_sum_w_hour_of_day_72_840_mean',
    'prod_act_sum_w_hour_of_day_72_840_std', 'prod_act_sum_w_hour_of_day_72_840_slope',
    'prod_act_sum_w_hour_of_weekday_4_mean', 'prod_act_sum_w_hour_of_weekday_4_std',
    'prod_act_sum_w_hour_of_weekday_4_slope', 'prod_act_sum_w_hour_of_weekday_8_mean',
    'prod_act_sum_w_hour_of_weekday_8_std', 'prod_act_sum_w_hour_of_weekday_8_slope',
    'prod_act_sum_w_hour_of_weekday_12_mean', 'prod_act_sum_w_hour_of_weekday_12_std',
    'prod_act_sum_w_hour_of_weekday_12_slope', 'missing_pinst_sum_lag_42_hours',
    'missing_pinst_sum_lag_43_hours', 'missing_pinst_sum_lag_44_hours',
    'missing_pinst_sum_lag_45_hours', 'missing_pinst_sum_lag_46_hours',
    'missing_pinst_sum_lag_47_hours', 'missing_pinst_sum_lag_48_hours',
    'missing_pinst_sum_lag_49_hours', 'missing_pinst_sum_lag_50_hours',
    'missing_pinst_sum_lag_51_hours', 'missing_pinst_sum_lag_52_hours',
    'missing_pinst_sum_lag_53_hours', 'missing_pinst_sum_lag_54_hours',
    'missing_pinst_sum_lag_55_hours', 'missing_pinst_sum_lag_56_hours',
    'missing_pinst_sum_lag_57_hours', 'missing_pinst_sum_lag_58_hours',
    'missing_pinst_sum_lag_59_hours', 'missing_pinst_sum_lag_60_hours',
    'missing_pinst_sum_lag_61_hours', 'missing_pinst_sum_lag_62_hours',
    'missing_pinst_sum_lag_63_hours', 'missing_pinst_sum_lag_64_hours',
    'missing_pinst_sum_lag_65_hours', 'missing_pinst_sum_lag_66_hours',
    'missing_pinst_sum_lag_67_hours', 'missing_pinst_sum_lag_68_hours',
    'missing_pinst_sum_lag_69_hours', 'missing_pinst_sum_lag_70_hours',
    'missing_pinst_sum_lag_71_hours', 'missing_pinst_sum_lag_72_hours',
    'missing_pinst_sum_lag_168_hours', 'missing_pinst_sum_w_hour_of_day_7_mean',
    'missing_pinst_sum_w_hour_of_day_7_std', 'missing_pinst_sum_w_hour_of_day_7_slope',
    'missing_pinst_sum_w_hour_of_day_14_mean', 'missing_pinst_sum_w_hour_of_day_14_std',
    'missing_pinst_sum_w_hour_of_day_14_slope', 'missing_pinst_sum_w_hour_of_day_28_mean',
    'missing_pinst_sum_w_hour_of_day_28_std', 'missing_pinst_sum_w_hour_of_day_28_slope',
    'missing_pinst_sum_w_hour_of_weekday_4_mean', 'missing_pinst_sum_w_hour_of_weekday_4_std',
    'missing_pinst_sum_w_hour_of_weekday_4_slope', 'missing_pinst_sum_w_hour_of_weekday_8_mean',
    'missing_pinst_sum_w_hour_of_weekday_8_std', 'missing_pinst_sum_w_hour_of_weekday_8_slope',
    'missing_pinst_sum_w_past_hours_42_48_mean', 'missing_pinst_sum_w_past_hours_42_48_std',
    'missing_pinst_sum_w_past_hours_42_48_slope', 'spot_price_act_lag_42_hours',
    'spot_price_act_lag_43_hours', 'spot_price_act_lag_44_hours',
    'spot_price_act_lag_45_hours', 'spot_price_act_lag_46_hours',
    'spot_price_act_lag_47_hours', 'spot_price_act_lag_48_hours',
    'spot_price_act_lag_168_hours', 'spot_price_act_w_hour_of_day_7_mean',
    'spot_price_act_w_hour_of_day_7_std', 'spot_price_act_w_hour_of_day_7_slope',
    'spot_price_act_w_hour_of_day_14_mean', 'spot_price_act_w_hour_of_day_14_std',
    'spot_price_act_w_hour_of_day_14_slope', 'spot_price_act_w_hour_of_day_28_mean',
    'spot_price_act_w_hour_of_day_28_std', 'spot_price_act_w_hour_of_day_28_slope',
    'spot_price_act_w_past_hours_42_48_mean', 'spot_price_act_w_past_hours_42_48_std',
    'spot_price_act_w_past_hours_42_48_slope', 'rebap_price_est_lag_42_hours',
    'rebap_price_est_lag_43_hours', 'rebap_price_est_lag_44_hours',
    'rebap_price_est_lag_45_hours', 'rebap_price_est_lag_46_hours',
    'rebap_price_est_lag_47_hours', 'rebap_price_est_lag_48_hours',
    'rebap_price_est_lag_168_hours', 'rebap_price_est_w_hour_of_day_7_mean',
    'rebap_price_est_w_hour_of_day_7_std', 'rebap_price_est_w_hour_of_day_7_slope',
    'rebap_price_est_w_hour_of_day_14_mean', 'rebap_price_est_w_hour_of_day_14_std',
    'rebap_price_est_w_hour_of_day_14_slope', 'rebap_price_est_w_hour_of_day_28_mean',
    'rebap_price_est_w_hour_of_day_28_std', 'rebap_price_est_w_hour_of_day_28_slope',
    'rebap_price_est_w_past_hours_42_48_mean', 'rebap_price_est_w_past_hours_42_48_std',
    'rebap_price_est_w_past_hours_42_48_slope', 'fcr_price_act_lag_42_hours',
    'fcr_price_act_lag_43_hours', 'fcr_price_act_lag_44_hours',
    'fcr_price_act_lag_45_hours', 'fcr_price_act_lag_46_hours',
    'fcr_price_act_lag_47_hours', 'fcr_price_act_lag_48_hours',
    'fcr_price_act_lag_168_hours', 'fcr_price_act_w_hour_of_day_7_mean',
    'fcr_price_act_w_hour_of_day_7_std', 'fcr_price_act_w_hour_of_day_7_slope',
    'fcr_price_act_w_hour_of_day_14_mean', 'fcr_price_act_w_hour_of_day_14_std',
    'fcr_price_act_w_hour_of_day_14_slope', 'fcr_price_act_w_hour_of_day_28_mean',
    'fcr_price_act_w_hour_of_day_28_std', 'fcr_price_act_w_hour_of_day_28_slope',
    'fcr_price_act_w_past_hours_42_48_mean', 'fcr_price_act_w_past_hours_42_48_std',
    'fcr_price_act_w_past_hours_42_48_slope', 'afrr_neg_avg_price_act_lag_42_hours',
    'afrr_neg_avg_price_act_lag_43_hours', 'afrr_neg_avg_price_act_lag_44_hours',
    'afrr_neg_avg_price_act_lag_45_hours', 'afrr_neg_avg_price_act_lag_46_hours',
    'afrr_neg_avg_price_act_lag_47_hours', 'afrr_neg_avg_price_act_lag_48_hours',
    'afrr_neg_avg_price_act_lag_168_hours', 'afrr_neg_avg_price_act_w_hour_of_day_7_mean',
    'afrr_neg_avg_price_act_w_hour_of_day_7_std', 'afrr_neg_avg_price_act_w_hour_of_day_7_slope',
    'afrr_neg_avg_price_act_w_hour_of_day_14_mean', 'afrr_neg_avg_price_act_w_hour_of_day_14_std',
    'afrr_neg_avg_price_act_w_hour_of_day_14_slope', 'afrr_neg_avg_price_act_w_hour_of_day_28_mean',
    'afrr_neg_avg_price_act_w_hour_of_day_28_std', 'afrr_neg_avg_price_act_w_hour_of_day_28_slope',
    'afrr_neg_avg_price_act_w_past_hours_42_48_mean', 'afrr_neg_avg_price_act_w_past_hours_42_48_std',
    'afrr_neg_avg_price_act_w_past_hours_42_48_slope', 'afrr_pos_avg_price_act_lag_42_hours',
    'afrr_pos_avg_price_act_lag_43_hours', 'afrr_pos_avg_price_act_lag_44_hours',
    'afrr_pos_avg_price_act_lag_45_hours', 'afrr_pos_avg_price_act_lag_46_hours',
    'afrr_pos_avg_price_act_lag_47_hours', 'afrr_pos_avg_price_act_lag_48_hours',
    'afrr_pos_avg_price_act_lag_168_hours', 'afrr_pos_avg_price_act_w_hour_of_day_7_mean',
    'afrr_pos_avg_price_act_w_hour_of_day_7_std', 'afrr_pos_avg_price_act_w_hour_of_day_7_slope',
    'afrr_pos_avg_price_act_w_hour_of_day_14_mean', 'afrr_pos_avg_price_act_w_hour_of_day_14_std',
    'afrr_pos_avg_price_act_w_hour_of_day_14_slope', 'afrr_pos_avg_price_act_w_hour_of_day_28_mean',
    'afrr_pos_avg_price_act_w_hour_of_day_28_std', 'afrr_pos_avg_price_act_w_hour_of_day_28_slope',
    'afrr_pos_avg_price_act_w_past_hours_42_48_mean', 'afrr_pos_avg_price_act_w_past_hours_42_48_std',
    'afrr_pos_avg_price_act_w_past_hours_42_48_slope',
]

DF_COLS = [
    'prod_act_sum', 'prod_fc_sum_lag_0_hours', 'temperature_fc_lag_0_hours',
    'solar_fc_lag_0_hours', 'wind_fc_lag_0_hours', 'prod_act_sum_lag_72_hours',
    'missing_pinst_sum_lag_42_hours', 'spot_price_act_lag_42_hours',
    'rebap_price_est_lag_42_hours', 'fcr_price_act_lag_42_hours',
    'afrr_neg_avg_price_act_lag_42_hours', 'afrr_pos_avg_price_act_lag_42_hours',
    'day_of_week_sin', 'day_of_week_cos', 'hour_of_day_sin', 'hour_of_day_cos',
    'day_of_year_sin', 'day_of_year_cos', 'weekend_or_holiday',
]


def generate_base_series(index, name):
    """Generate base time series with realistic patterns."""
    if name == 'prod_act_sum':
        base = 740 + 150 * np.sin(2 * np.pi * np.arange(len(index)) / 24)
        weekly = 50 * np.sin(2 * np.pi * np.arange(len(index)) / 168)
        noise = np.random.normal(0, 30, len(index))
        series = base + weekly + noise
        return np.clip(series, 0, 1500)

    elif name == 'prod_fc_sum':
        base = 740 + 150 * np.sin(2 * np.pi * np.arange(len(index)) / 24)
        weekly = 50 * np.sin(2 * np.pi * np.arange(len(index)) / 168)
        noise = np.random.normal(0, 40, len(index))
        series = base + weekly + noise
        return np.clip(series, 0, 1500)

    elif name == 'temperature_fc':
        day_of_year = index.dayofyear.values
        seasonal = 12 + 10 * np.cos(2 * np.pi * day_of_year / 365)
        hourly_noise = np.random.normal(0, 1, len(index))
        return seasonal + hourly_noise

    elif name == 'solar_fc':
        hour = index.hour.values
        day_of_year = index.dayofyear.values
        daily_pattern = np.where((hour >= 6) & (hour <= 20),
                                 np.sin(np.pi * (hour - 6) / 14) * 600, 0)
        seasonal_amp = 0.2 + 0.8 * (0.5 + 0.5 * np.sin(2 * np.pi * day_of_year / 365))
        series = daily_pattern * seasonal_amp
        return np.clip(series, 0, 800)

    elif name == 'wind_fc' or name == 'log_wind_fc':
        series = np.cumsum(np.random.normal(-1, 10, len(index))) + 200
        series = np.clip(series, 0, 800)
        if name == 'log_wind_fc':
            return np.log1p(series)
        return series

    elif name in ['rebap_price_est', 'spot_price_act', 'fcr_price_act']:
        center = {'rebap_price_est': 50, 'spot_price_act': 60, 'fcr_price_act': 20}[name]
        series = center + np.cumsum(np.random.normal(0, 1, len(index)))
        return series

    elif name in ['afrr_neg_avg_price_act', 'afrr_pos_avg_price_act']:
        series = 15 + np.cumsum(np.random.normal(0, 0.5, len(index)))
        return np.clip(series, 0, 200)

    elif name == 'missing_pinst_sum':
        series = 50 + np.cumsum(np.random.normal(0, 0.1, len(index)))
        return np.clip(series, 0, 200)

    elif name == 'pinst_active_sum':
        return 1200 + np.random.normal(0, 20, len(index))

    return None


def create_lag_column(base_series, lag):
    """Create a lagged column."""
    return base_series.shift(lag)


def create_window_column(base_series, window_size, agg_type):
    """Create a window aggregation column."""
    if agg_type == 'mean':
        return base_series.rolling(window=window_size, min_periods=1).mean()
    elif agg_type == 'std':
        return base_series.rolling(window=window_size, min_periods=1).std()
    elif agg_type == 'slope':
        def slope(x):
            if len(x) < 2:
                return np.nan
            return (x.iloc[-1] - x.iloc[0]) / len(x)
        return base_series.rolling(window=window_size, min_periods=1).apply(slope)
    return None


def generate_dummy_training_data():
    """Generate cached_training_data.pkl."""
    index = pd.date_range('2022-02-01', '2026-01-01', freq='h', tz='CET')
    data = {}

    # Base series
    bases = {
        'prod_act_sum': generate_base_series(index, 'prod_act_sum'),
        'prod_fc_sum': generate_base_series(index, 'prod_fc_sum'),
        'temperature_fc': generate_base_series(index, 'temperature_fc'),
        'solar_fc': generate_base_series(index, 'solar_fc'),
        'log_wind_fc': generate_base_series(index, 'log_wind_fc'),
        'rebap_price_est': generate_base_series(index, 'rebap_price_est'),
        'spot_price_act': generate_base_series(index, 'spot_price_act'),
        'rebap_price_est': generate_base_series(index, 'rebap_price_est'),
        'fcr_price_act': generate_base_series(index, 'fcr_price_act'),
        'afrr_neg_avg_price_act': generate_base_series(index, 'afrr_neg_avg_price_act'),
        'afrr_pos_avg_price_act': generate_base_series(index, 'afrr_pos_avg_price_act'),
        'missing_pinst_sum': generate_base_series(index, 'missing_pinst_sum'),
        'pinst_active_sum': generate_base_series(index, 'pinst_active_sum'),
    }

    # Convert to Series for easier handling
    for k, v in bases.items():
        bases[k] = pd.Series(v, index=index)

    # Generate all columns
    for col in CACHED_TRAINING_DATA_COLS:
        if col in bases:
            data[col] = bases[col].values
        else:
            # Parse lag/window columns
            if '_lag_' in col:
                base_name = col.split('_lag_')[0]
                lag = int(col.split('_lag_')[1].split('_')[0])
                if base_name in bases:
                    data[col] = create_lag_column(bases[base_name], lag).values
            elif '_w_' in col:
                parts = col.split('_w_')
                base_name = parts[0]
                rest = parts[1]

                if 'hour_of_day_' in rest:
                    size = int(rest.split('hour_of_day_')[1].split('_')[0])
                    agg = rest.split('_')[-1]
                    if base_name in bases:
                        data[col] = create_window_column(bases[base_name], size, agg).values
                elif 'hour_of_weekday_' in rest:
                    size = int(rest.split('hour_of_weekday_')[1].split('_')[0]) * 24
                    agg = rest.split('_')[-1]
                    if base_name in bases:
                        data[col] = create_window_column(bases[base_name], size, agg).values
                elif 'past_hours_' in rest:
                    range_str = rest.split('past_hours_')[1].split('_')[0:2]
                    start, end = int(range_str[0]), int(range_str[1])
                    agg = rest.split('_')[-1]
                    if base_name in bases:
                        size = end - start
                        series = create_window_column(bases[base_name], size, agg)
                        data[col] = series.shift(start).values if series is not None else None

            if col not in data or data[col] is None:
                data[col] = np.random.normal(0, 10, len(index))

    df = pd.DataFrame(data, index=index)
    df = df[CACHED_TRAINING_DATA_COLS]  # Ensure correct column order
    return df


def main():
    here = Path(__file__).parent

    print("Generating cached_training_data.pkl...")
    df_train = generate_dummy_training_data()
    df_train.to_pickle(here / 'cached_training_data.pkl')
    print(f"  Shape: {df_train.shape}, TZ: {df_train.index.tz}")

    print("\nDummy data generation complete!")


if __name__ == '__main__':
    main()
