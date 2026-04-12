# src/predict.py
import pandas as pd
import numpy as np
import joblib
import os
from src.feature_engineering import add_features

def prepare_inference_data(train_aggregated, test_path, config):
    # 1. Gestion du nom de fichier
    if not os.path.exists(test_path):
        test_path = "data/raw/test .csv" if os.path.exists("data/raw/test .csv") else test_path

    # 2. Charger le test
    test_raw = pd.read_csv(test_path, parse_dates=['date'])
    test_daily = test_raw[test_raw['store_nbr'] == config['params']['store_nbr']]
    test_daily = test_daily.groupby('date').agg({'onpromotion': 'sum'}).reset_index()
    test_daily['sales'] = np.nan 

    # --- AJOUT SÉCURITÉ : HOLIDAYS ---
    # On charge les jours fériés pour le futur aussi
    holidays = pd.read_csv(config['paths']['raw_holidays'], parse_dates=['date'])
    holidays = holidays[holidays['transferred'] == False]
    test_daily['is_holiday'] = test_daily['date'].isin(holidays['date']).astype(int)
    # ---------------------------------

    # 3. Récupérer l'historique (Train)
    max_lag = max(config['params']['lags'])
    history = train_aggregated.tail(max_lag + 10).copy()
    
    # 4. Concaténation
    combined = pd.concat([history, test_daily], axis=0).reset_index(drop=True)
    
    # 5. Calcul des features
    combined_with_features = add_features(combined, 
                                          lags=config['params']['lags'], 
                                          window=config['params']['rolling_window'])
    
    # 6. Isoler les lignes du futur
    first_test_date = test_daily['date'].min()
    test_ready = combined_with_features[combined_with_features['date'] >= first_test_date].copy()
    
    print(f"DEBUG: Nombre de lignes prêtes pour la prédiction : {len(test_ready)}")
    return test_ready

def run_prediction(test_ready, model_package_path):
    package = joblib.load(model_package_path)
    model = package['model']
    features = package['feature_names']
    
    # --- LA CORRECTION CRITIQUE ICI ---
    # 1. On s'assure que toutes les colonnes attendues existent
    for col in features:
        if col not in test_ready.columns:
            test_ready[col] = 0
    
    # 2. On remplace TOUS les NaN restants par 0 (ou la moyenne) 
    # pour éviter le crash du modèle
    X_test = test_ready[features].fillna(0)
    
    # 3. Prédiction
    test_ready['predicted_sales'] = model.predict(X_test)
    
    return test_ready[['date', 'predicted_sales']]