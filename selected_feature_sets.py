"""Recommended feature sets exported from the `feature_selection_funnel`
MLflow experiment (runs tagged `selected_feature_sets`). Each list is the
`recommended_features` payload from a run's `sfs_recommended_features.json`
artifact, named `<model_type_sfs>[_<experiment>]`."""

catboost_restricted = [
    "pinst_active_sum_lag_0_hours",
    "temperature_fc_w_past_hours_0_48_mean",
    "solar_fc_w_hour_of_day_14_mean",
    "hour_of_day_sin",
    "hour_of_day_cos",
    "day_of_week_sin",
]

catboost_main = [
    "prod_fc_sum_lag_0_hours",
    "prod_fc_sum_w_hour_of_day_7_mean",
    "prod_act_sum_w_hour_of_day_72_168_mean",
]
