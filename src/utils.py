# utils.py - Fonctions utilitaires pour le projet de prévision des ventes   
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def save_model_pipeline(model, feature_names, metrics, output_path='models/model_production.pkl'):
    model_package = {
        'model': model,
        'feature_names': feature_names,
        'metrics': metrics,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': type(model).__name__,
        'version': '1.0.0'
    }
    joblib.dump(model_package, output_path)
    print(f"✅ Modèle sauvegardé : {output_path}")
    return model_package

def plot_feature_importance(model, feature_names, output_path='outputs/features.png'):
    if not hasattr(model, 'feature_importances_'): return
    
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_})
    feat_imp = feat_imp.sort_values('Importance', ascending=True).tail(10)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feat_imp['Feature'], feat_imp['Importance'])
    plt.title('Top 10 Features')
    plt.savefig(output_path)
    plt.close()

def export_for_powerbi(df_original, predictions, model_name, output_path):
    export_df = df_original[['date']].copy()
    export_df['sales_actual'] = df_original['sales']
    export_df['sales_predicted'] = predictions
    export_df['model_name'] = model_name
    export_df['export_date'] = pd.Timestamp.now()
    export_df.to_csv(output_path, index=False, encoding='utf-8-sig')