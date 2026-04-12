"""Script principal pour prédire les ventes futures à partir du fichier test.csv.
Il s'agit d'une extension du script de prédiction classique, avec une attention particulière à la préparation des données d'inférence pour le test.csv, en s'assurant que les lags et autres features temporelles sont correctement calculés."""
import yaml
from src.preprocessing import clean_and_merge
from src.predict import prepare_inference_data, run_prediction

def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. On a besoin du train pour avoir les "ventes de la veille"
    df_train_clean = clean_and_merge(config['paths']['raw_train'], 
                                     config['paths']['raw_transactions'], 
                                     config['paths']['raw_holidays'])

    # 2. Préparation des données de test avec la suture temporelle
    test_features = prepare_inference_data(df_train_clean, config['paths']['raw_test'], config)

    # 3. Inférence
    results = run_prediction(test_features, config['paths']['model_path'])

    # 4. Sauvegarde
    results.to_csv("outputs/final_test_predictions.csv", index=False)
    print("✅ Prédictions pour test.csv générées avec succès !")

if __name__ == "__main__":
    main()