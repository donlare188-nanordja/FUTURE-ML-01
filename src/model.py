"""
On implémente la logique de comparaison automatique pour déterminer quel algorithme est le plus performant sur vos données de ventes
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate(y_true, y_pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true[y_true != 0])) * 100
    }

def compare_models(X, y, n_splits=5):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for name, model in models.items():
        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])
            cv_scores.append(evaluate(y.iloc[val_idx], preds)['RMSE'])
        
        results.append({'Model': name, 'Mean_RMSE': np.mean(cv_scores), 'Model_Obj': model})
    
    results_df = pd.DataFrame(results).sort_values('Mean_RMSE')
    return results_df.iloc[0], results_df