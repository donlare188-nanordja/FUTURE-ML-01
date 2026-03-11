# FUTURE-ML-01

Construire un modèle de ML qui prend des données historique de ventes , prévoit les ventes  futures , utilise power bi pour la visualisation et donne une interprétation business .

# 📊 Sales Forecasting - Machine Learning Project

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2.0-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Prévision des ventes par séries temporelles** - Modèle ML déployable en production pour l'optimisation des stocks et la planification commerciale.

---

## 📋 Table des Matières

- [Vue d'ensemble](#vue-densemble)
- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
- [Utilisation Rapide](#utilisation-rapide)
- [Structure du Projet](#structure-du-projet)
- [Performance du Modèle](#performance-du-modèle)
- [Déploiement en Production](#déploiement-en-production)
- [Documentation](#documentation)
- [Auteur](#auteur)
- [License](#license)

---

## 🎯 Vue d'ensemble

Ce projet développe un système de prévision des ventes basé sur l'apprentissage automatique pour optimiser la gestion des stocks et améliorer la planification commerciale.

### Problématique Business

- **Réduire les ruptures de stock** grâce à des prévisions précises
- **Optimiser les niveaux de stock** pour diminuer les coûts de stockage
- **Anticiper la demande** hebdomadaire avec une précision de >90%

### Solution Technique

- **Algorithmes testés** : Random Forest, XGBoost, Gradient Boosting, Linear Regression
- **Validation** : Time Series Cross-Validation (5 folds)
- **Métriques** : RMSE, MAE, MAPE
- **Feature Engineering** : Lag features, Rolling statistics, Variables temporelles

---

## ✨ Fonctionnalités

### ✅ Core Features

- [x] Comparaison automatique de 4 algorithmes ML
- [x] Feature Engineering avancé (lags, rolling means)
- [x] Time Series Split pour validation temporelle
- [x] Analyse des Feature Importances
- [x] Visualisation Prédictions vs Réalité
- [x] Export automatisé pour Power BI
- [x] Sauvegarde du modèle prêt pour production

### 🚀 Production Ready

- [x] Script de prédiction autonome
- [x] Gestion des versions de modèles
- [x] Logging et monitoring
- [x] Documentation complète
- [x] API REST (à faire)

---

## 📦 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de packages Python)
- Git

### Étapes d'installation

1. **Cloner le repository**
   
   ```bash
   git clone https://github.com/votre-username/sales-forecasting.git
   cd sales-forecasting
   ```

2. **Créer un environnement virtuel (recommandé)**

pip install -r requirements.txt

pip install -r requirements.txt

4. **Vérifier l'installation**

python -c "import sklearn; print(f'Scikit-Learn {sklearn.__version__}')"

## ⚡ Utilisation Rapide

### 1. Entraînement du Modèle

Exécuter le notebook Jupyter pour l'entraînement :

jupyter forecasting.ipynb

Ou exécuter le script d'entraînement :

python 01_train_model.py

**Résultats attendus :**

- Génération du fichier `model_production.pkl`
- Affichage des métriques (RMSE, MAE, MAPE)
- Graphiques de feature importance
- Visualisation des prédictions

### 2. Prédiction en Production

Pour générer des prédictions quotidiennes :

python 02_production_script.py

**Ce script va :**

- Charger le modèle entraîné
- Préparer les features automatiquement
- Générer les prédictions pour le lendemain
- Sauvegarder les résultats dans `predictions_output.csv`

### 3. Visualisation des Résultats

Ouvrir Power BI et importer le fichier `predictions_powerbi.csv` pour visualiser :

- Prédictions vs Ventes réelles
- Analyse des erreurs (MAPE)
- Tendances temporelles

FUTURE-ML-01/
├── 📄 .gitignore # Fichiers ignorés par Git
├── 📄 README.md # Ce fichier
├── 📄 requirements.txt # Dépendances Python
├── 📁 Script/ # Scripts de production
│ ├── 📄 02_production_script.py # Prédiction quotidienne
│ └── 📄 03_api_app.py # API FastAPI
├── 📁 notebook/ # Notebooks de développement
│ └── 📄 forecasting.ipynb # Entraînement & évaluation
├── 📁 powerbi/ # Export pour Power BI
│ └── 📄 predictions_powerbi.csv # Prédictions formatées
├── 📁 data/ # ❌ Exclus de Git (trop lourds)
│ └── 📄 [Sur Google Drive] # 🔗 Lien dans section ci-dessous
└── 📁 model/ # ❌ Exclus de Git (trop lourds)
└── 📄 [Sur Google Drive] # 🔗 Lien dans section ci-dessous

## 📊 Performance du Modèle

### Métriques sur le Test Set (20% des données)

### Métriques sur le Test Set (20% des données)

| Métrique | Valeur   | Interprétation                         |
| -------- | -------- | -------------------------------------- |
| **RMSE** | 1,459.56 | Erreur quadratique moyenne             |
| **MAE**  | 1,150.34 | Erreur absolue moyenne (~1,150 ventes) |
| **MAPE** | 9.85%    | Précision de **90.15%** ✅              |

### Comparaison des Algorithmes

| Modèle            | RMSE     | MAE      | MAPE   | Rang    |
| ----------------- | -------- | -------- | ------ | ------- |
| **Random Forest** | 1,459.56 | 1,150.34 | 9.85%  | 🥇 1er  |
| Gradient Boosting | 1,528.36 | 1,205.67 | 10.42% | 🥈 2ème |
| Linear Regression | 1,600.07 | 1,291.56 | 11.26% | 🥉 3ème |
| XGBoost           | 1,687.09 | 1,350.89 | 11.89% | 4ème    |

### Feature Importances (Top 5)

1. **lag_7** (51.87%) - Saisonnalité hebdomadaire
2. **transactions** (23.26%) - Volume de transactions
3. **rolling_mean_7** (11.28%) - Tendance récente
4. **onpromotion** (6.55%) - Impact des promotions
5. **lag_1** (2.12%) - Ventes de la veille

---

## 🚀 Déploiement en Production

### Architecture Recommandée

┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Base de   │────▶│   Script     │────▶│   Fichier   │
│   Données   │     │  Python      │     │   CSV       │
│   (SQL)     │     │ (Quotidien)  │     │             │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Power BI   │
                    │  Dashboard   │
                    └──────────────┘

### Planification Automatisée

**Windows Task Scheduler :**

# Exécuter tous les jours à 02:00

0 2 * * * python C:\chemin\vers\02_production_script.py

### Monitoring

Surveiller quotidiennement :

- ✅ **MAPE** : Doit rester < 15%
- ✅ **Logs** : Vérifier `prediction_log.log`
- ✅ **Data Drift** : Comparer distributions des features

### Ré-entraînement

**Fréquence recommandée :** Mensuelle

# 1. Ajouter les nouvelles données

python scripts/update_data.py

# 2. Ré-entraîner le modèle

jupyter nbconvert --to notebook --execute 01_Training_Notebook.ipynb

# 3. Versionner le nouveau modèle

cp models/model_production.pkl models/model_archive/model_v1.0.1.pkl
cp models/model_new.pkl models/model_production.pkl

### Ressources Externes

- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [Time Series Forecasting Guide](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
- [XGBoost Parameters](https://xgboost.readthedocs.io/)

---

## 👨‍💻 Auteur

**Dieudonné [LARE N.Donné]**

- 📧 Email: donlare188@gmail.com
- 💼 LinkedIn: [[linkedin.com/in/don-lare-4491693b1](https://www.linkedin.com/in/don-lare-4491693b1)](https://linkedin.com/)
- 🐙 GitHub: https://github.com/donlare188-nanordja/FUTURE-ML-01.git](https://github.com/)

**Projet réalisé dans le cadre d'un internship en Machine Learning

---

## 🙏 Remerciements

- **Kaggle** pour le dataset ["Store Sales - Time Series Forecasting"](https://www.kaggle.com/c/store-sales-time-series-forecasting?spm=a2ty_o01.29997173.0.0.43b15171Y5D67L)
- **Scikit-Learn** et **XGBoost** pour les bibliothèques ML
- **Mentors et collègues** pour leur soutien et leurs conseils

---

## 📬 Contact

Pour toute question ou collaboration :

- Ouvrir une **issue** sur GitHub
- Envoyer un **email** à [donlare188@gmail.com](mailto:donlare188@gmail.com)

---

<div align="center">

**⭐ Si ce projet vous a été utile, n'hésitez pas à mettre une étoile !** ⭐

Made with ❤️ by Dieudonné

</div>
