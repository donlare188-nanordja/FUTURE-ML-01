#  ce fichier est dédié à l'entraînement du modèle de production, en utilisant les fonctions de comparaison et d'analyse développées précédemment.
from src.model import compare_models
from src.utils import plot_feature_importance, save_model_pipeline

def train_production_model(X, y, config):
    print(" Comparaison des modèles en cours...")
    best_info, _ = compare_models(X, y)
    best_model = best_info['Model_Obj']
    
    print(f" Meilleur modèle sélectionné : {best_info['Model']}")
    
    # Ré-entraînement final sur l'intégralité des données
    best_model.fit(X, y)
    
    # Analyse et Sauvegarde
    plot_feature_importance(best_model, X.columns.tolist())
    
    return best_model, best_info['Model']