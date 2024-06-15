import pandas as pd
import numpy as np
import holidays
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from dowhy import CausalModel

def train_model(train_df, test_df, features):
    X_train = train_df[features]
    y_train = train_df['unfulfilled_requests']
    X_test = test_df[features]
    y_test = test_df['unfulfilled_requests']

    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        results[name] = mse

    return results