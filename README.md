# Projet de session — Forage des données (GUIDE)

## Contexte
Les SOC (Security Operations Centers) font face à une explosion du volume d’alertes. Cette surcharge complique le triage rapide et fiable des incidents. Le dataset **GUIDE** de Microsoft fournit une base réaliste et massive d’incidents afin de construire des modèles capables d’assister les analystes dans la priorisation.

## Problématique
Comment automatiser la **classification des incidents de sécurité** (TP vs Non‑TP) à grande échelle, tout en conservant des performances robustes malgré :
- un **volume massif** de données,
- des **variables hétérogènes** et à forte cardinalité,
- un **risque de surapprentissage** ?

## Objectif
Ce projet vise à classifier les incidents de cybersécurité issus du dataset **GUIDE (Microsoft Security Incident Prediction)** en deux classes :
- `TruePositive` (TP)
- `Non-TruePositive` (0 = Benign/False)

Il s’appuie sur un pipeline de prétraitement complet (notebook) et un script d’entraînement centralisé (`Modeles.py`) qui exporte les modèles prêts pour l’application Streamlit.

## Contenu du notebook (`notebook.ipynb`)
Le notebook contient tout le **prétraitement** et la **préparation des données** :

1. **Chargement & échantillonnage stratifié**
	- Lecture par blocs d’un fichier massif (GUIDE ~2.4 Go)
	- Échantillon stratifié (1%) conservant la distribution de `IncidentGrade`

2. **Nettoyage & traitement des NaN**
	- Remplacement intelligent (`Unknown`, `None`) selon le sens métier
	- Suppression des lignes sans cible

3. **Réduction de cardinalité**
	- `AlertTitle`, `ThreatFamily`, `CountryCode` regroupées en top N + `Other`
	- Extraction des techniques MITRE principales

4. **Feature engineering temporel**
	- `Month`, `IsWeekend`, `IsBusinessHour`, `DayOfWeek`, `Hour`, etc.

5. **Encodage & exports**
	- `donnees_preprocessees.csv` (avant OHE)
	- `donnees_transformees.npz` (après StandardScaler + OneHotEncoder)
	- `preprocessor.joblib` (pipeline de transformation)

Ces fichiers sont ensuite utilisés directement par `Modeles.py`.

## Entraînement des modèles (`Modeles.py`)
Le script centralise tout l’entraînement + évaluation + export des modèles :

### Modèles entraînés
1. **Régression logistique + PCA**
	- Optimisée via `RandomizedSearchCV`

2. **KNN + PCA**
	- `Elbow Method` + `GridSearchCV`

3. **Random Forest**
	- `RandomizedSearchCV`
	- Sélection de features par importance cumulée
	- Optimisation du **seuil de décision** (threshold tuning)

### Exports générés
Le script crée le dossier `models/` avec :
- `best_regression.joblib`
- `best_knn.joblib`
- `rf_model.joblib`
- `pca.joblib`
- `preprocessor.joblib`
- `rf_preprocessor.joblib`
- `reg_metadata.json`, `knn_metadata.json`, `rf_metadata.json`

Ces fichiers sont directement consommés par `app.py`.

## Application Streamlit (`app.py`)
L’interface Streamlit charge **uniquement** les fichiers exportés dans `models/`.
Elle ne ré-exécute pas `Modeles.py`.

Fonctionnalités principales :
- Analyse d’un fichier CSV
- Analyse manuelle via formulaire (valeurs préremplies)
- Visualisations : KPI, distributions, ROC / PR

**Note** : une vidéo de démonstration sera faite plus tard (par l’utilisateur).

## Structure des fichiers importants
- `notebook.ipynb` → prétraitement & exports
- `Modeles.py` → entraînement + export des modèles
- `models/` → modèles et metadata utilisés par l’app
- `donnees_preprocessees.csv` → dataset avant encodage
- `donnees_transformees.npz` → dataset transformé (train/test)
- `preprocessor.joblib` → pipeline de transformation

## Comment utiliser le pipeline
1. Exécuter `notebook.ipynb` pour générer les données transformées.
2. Lancer `Modeles.py` pour entraîner et exporter les modèles.
3. Lancer `app.py` pour utiliser l’interface Streamlit.

Si tu veux, je peux ajouter un schéma de flux ou un mini guide d’exécution pas-à-pas.

