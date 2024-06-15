import pandas as pd
import numpy as np
from dowhy import CausalModel

df = pd.read_csv('../data/processeddata/mergeddata.csv')

train_df = df.sample(frac=0.8, random_state=42)
holdout_df = df.drop(train_df.index)


def perform_causal_analysis(train_df):
    # Define interventions
    interventions = {
        'driver_movement': 1,
        'order_location_accuracy': 5,
        'driver_operating_time': 8,
        'num_drivers': 10
    }

    results = {}

    for treatment in interventions.keys():
        model = CausalModel(
            data=train_df,
            treatment=[treatment],
            outcome='unfulfilled_requests',
            common_causes=['hour', 'day_of_week', 'is_weekend', 'driver_id', 'duration_hours', 'holiday']
        )

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        causal_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression"
        )

        results[treatment] = causal_estimate.value

    return results