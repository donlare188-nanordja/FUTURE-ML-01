""" 
Script principal pour l'entraînement du modèle de prévision des ventes 
et l'export des prédictions pour Power BI.
"""

import yaml
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Imports locaux
from src.preprocessing import clean_and_merge
from src.feature_engineering import add_features
from src.train import train_production_model
from src.utils import export_for_powerbi, save_model_pipeline

def main():
    # 0. Chargement de la configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Création du dossier models s'il n'existe pas
    os.makedirs(os.path.dirname(config['paths']['model_path']), exist_ok=True)

    # 1. Préparation des données (Train)
    print(" Chargement et nettoyage des données...")
    data = clean_and_merge(config['paths']['raw_train'], 
                           config['paths']['raw_transactions'], 
                           config['paths']['raw_holidays'])
    
    print(f"Columns: {data.columns.tolist()}")
    
    df_model = add_features(data)
    X = df_model.drop(columns=['date', 'sales'])
    y = df_model['sales']

    # 2. Entraînement et Sélection du meilleur modèle
    best_model, model_name = train_production_model(X, y, config)
    
    # 3. Calcul des prédictions sur le set d'entraînement pour l'export
    final_preds = best_model.predict(X)

    # 4. Calcul des métriques finales
    final_metrics = {
        'RMSE': float(np.sqrt(mean_squared_error(y, final_preds))),
        'MAE': float(mean_absolute_error(y, final_preds)),
        'MAPE': float(np.mean(np.abs((y - final_preds) / y[y != 0])) * 100)
    }

    # 5. Sauvegarde du modèle pour l'inférence future (test.csv)
    print(f" Sauvegarde du modèle dans {config['paths']['model_path']}...")
    save_model_pipeline(
        model=best_model, 
        feature_names=X.columns.tolist(), 
        metrics=final_metrics, 
        output_path=config['paths']['model_path']
    )

    # 6. Export pour Power BI
    print(f" Export des résultats vers {config['paths']['output_powerbi']}...")
    export_for_powerbi(df_model, final_preds, model_name, config['paths']['output_powerbi'])

    print(f"\n Terminé ! Modèle prêt et fichier Power BI généré.")
    print(f"   Moyenne RMSE: {final_metrics['RMSE']:.2f}")

if __name__ == "__main__":
    main()