"""Module de feature engineering pour la préparation des données avant l'entraînement du modèle.
Il contient des fonctions pour créer des features temporelles, des lags et des moyennes mobiles, qui sont essentielles pour capturer les tendances et les patterns dans les données de séries temporelles."""

import pandas as pd

def add_features(df, lags=[1, 7, 14, 30], window=7):
    df = df.copy()
    
    # 1. Features Temporelles
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    
    # 2. Création des Lags
    created_lags = []
    for lag in lags:
        col_name = f'lag_{lag}'
        df[col_name] = df['sales'].shift(lag)
        created_lags.append(col_name)
        
    # 3. Moyenne mobile
    rolling_col = f'rolling_mean_{window}'
    df[rolling_col] = df['sales'].shift(1).rolling(window=window).mean()
    
    # --- FILTRAGE ROBUSTE ---
    # Au lieu de chercher 'lag_30', on cherche le lag le plus lointain
    # qui a été effectivement créé dans la boucle ci-dessus.
    max_lag_col = f'lag_{max(lags)}'
    
    if max_lag_col in df.columns:
        # On garde les lignes où le plus grand lag est rempli (données historiques)
        # OU les lignes où sales est NaN (données futures du test.csv)
        mask = (~df[max_lag_col].isna()) | (df['sales'].isna())
        df = df[mask].copy()
    
    return df