# Projet de session — Forage des données (GUIDE)

## Membres de l'équipe : Oscar Neveux | Wilson Fotsing | Guy Junior Calvet | Jean-Christophe Barriault

## Contexte
Les SOC (Security Operations Centers) font face à une explosion du volume d'alertes. Cette surcharge complique le triage rapide et fiable des incidents. Le dataset **GUIDE** de Microsoft fournit une base réaliste et massive d'incidents afin de construire des modèles capables d'assister les analystes dans la priorisation.

## Problématique
Comment automatiser la **classification des incidents de sécurité** (TP vs Non-TP) à grande échelle, tout en conservant des performances robustes malgré :
- un **volume massif** de données,
- des **variables hétérogènes** et à forte cardinalité,
- un **risque de surapprentissage** ?

## Objectif
Ce projet vise à classifier les incidents de cybersécurité issus du dataset **GUIDE (Microsoft Security Incident Prediction)** en deux classes :
- `TruePositive` (TP)
- `Non-TruePositive` (0 = Benign/False)

Il s'appuie sur un pipeline de prétraitement complet (`notebook.ipynb`) et un notebook d'entraînement documenté (`Modeles.ipynb`) qui exporte les modèles prêts pour l'application Streamlit.

Le **Random Forest** est retenu comme modèle de production : il obtient le meilleur rappel (moins d'incidents manqués), ce qui est critique en cybersécurité.

## Contenu du notebook (`notebook.ipynb`)
Le notebook contient tout le **prétraitement** et la **préparation des données** :

1. **Chargement & échantillonnage stratifié**
	- Lecture par blocs d'un fichier massif (GUIDE ~2.4 Go)
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

Ces fichiers sont ensuite utilisés directement par `Modeles.ipynb`.

## Entraînement des modèles (`Modeles.ipynb`)
Ce notebook documenté contient l'entraînement et l'évaluation des modèles. Chaque étape est accompagnée d'explications sur les choix méthodologiques (métriques, hyperparamètres, PCA vs non-PCA).

### Modèles entraînés
1. **Régression logistique + PCA**
	- Baseline sans/avec PCA
	- Optimisée via `RandomizedSearchCV`

2. **KNN + PCA**
	- Baseline sans/avec PCA
	- `Elbow Method` + `GridSearchCV`

3. **Random Forest (sans PCA)**
	- `RandomizedSearchCV` (n_iter=50)
	- Sélection de features par importance cumulée (seuils 80-95%)
	- Optimisation du **seuil de décision** (threshold tuning)
	- Justification de l'absence de PCA : robustesse native, interprétabilité des features, sélection par importance supérieure à la PCA

### Analyse des erreurs
- Matrices de confusion côte à côte pour les 3 modèles
- Focus sur les faux négatifs (incidents manqués) — erreur la plus critique en cybersécurité
- **Conclusion** : le Random Forest est retenu pour la mise en production (meilleur rappel, moins d'incidents manqués)

### Exports générés
Le notebook crée le dossier `models/` avec :
- `best_regression.joblib`
- `best_knn.joblib`
- `rf_model.joblib`
- `pca.joblib`
- `reg_metadata.json`, `knn_metadata.json`, `rf_metadata.json`

Ces fichiers sont directement consommés par `app.py`.

## Application Streamlit (`app.py`)
L'interface Streamlit charge **uniquement** les fichiers exportés dans `models/`.
Elle ne ré-exécute pas `Modeles.ipynb`.

Fonctionnalités principales :
- Analyse d'un fichier CSV
- Analyse manuelle via formulaire (valeurs préremplies)
- Visualisations : KPI, distributions, ROC / PR

## Structure des fichiers importants
- `notebook.ipynb` — prétraitement & exports
- `Modeles.ipynb` — entraînement documenté + export des modèles
- `app.py` — application Streamlit
- `models/` — modèles et metadata utilisés par l'app
- `donnees_preprocessees.csv` — dataset avant encodage
- `donnees_transformees.npz` — dataset transformé (train/test)
- `preprocessor.joblib` — pipeline de transformation

## Comment utiliser le pipeline
1. Exécuter `notebook.ipynb` pour générer les données transformées.
2. Exécuter `Modeles.ipynb` pour entraîner et exporter les modèles.
3. Lancer `streamlit run app.py` pour utiliser l'interface.
