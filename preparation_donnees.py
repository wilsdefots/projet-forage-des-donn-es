"""
Module de préparation des données — Projet de forage des données
================================================================

Ce module centralise toutes les étapes de prétraitement du dataset GUIDE
(Microsoft Security Incident Prediction), identiques à celles validées
sur la branche `main`.

Étapes :
    1. Sélection des colonnes utiles
    2. Nettoyage des valeurs manquantes
    3. Simplification de MitreTechniques (haute cardinalité)
    4. Réduction de cardinalité des colonnes géographiques
    5. Feature engineering temporel + binarisation
    6. Encodage de la variable cible
    7. Séparation X / y
    8. Split train/test stratifié
    9. Pipeline de preprocessing (StandardScaler + OneHotEncoder)

Références :
    - Scikit-learn ColumnTransformer :
      https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
    - Scikit-learn Pipeline :
      https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    - Dataset GUIDE :
      https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction

Auteur : Guy Junior Calvet
Branche : Guyjc-Modélisation
"""

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# ─────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────
COLONNES_UTILES = [
    'Timestamp',
    'Category',
    'MitreTechniques',
    'IncidentGrade',       # variable cible
    'EntityType',
    'EvidenceRole',
    'RegistryValueData',   # sera traitée en catégorielle
    'OSFamily',
    'OSVersion',
    'SuspicionLevel',
    'LastVerdict',
    'CountryCode',
    'State',
    'City',
]

CIBLE_MAP = {'BenignPositive': 0, 'FalsePositive': 1, 'TruePositive': 2}
CIBLE_MAP_INV = {v: k for k, v in CIBLE_MAP.items()}


# ─────────────────────────────────────────────────────────────
# Fonctions utilitaires
# ─────────────────────────────────────────────────────────────
def extraire_technique_principale(val: str) -> str:
    """Extrait le premier identifiant MITRE T-xxxx d'une chaîne.

    Parameters
    ----------
    val : str
        Valeur brute de la colonne MitreTechniques.

    Returns
    -------
    str
        Identifiant MITRE (ex. 'T1059.001') ou 'Unknown'/'Other'.
    """
    if pd.isna(val) or val == 'Unknown':
        return 'Unknown'
    match = re.search(r'T\d{4}(?:\.\d{3})?', str(val))
    return match.group(0) if match else 'Other'


def reduire_cardinalite(series: pd.Series, top_n: int = 20,
                        fill: str = 'Other') -> pd.Series:
    """Remplace les valeurs hors top_n par *fill*.

    Parameters
    ----------
    series : pd.Series
        Colonne catégorielle à réduire.
    top_n : int
        Nombre de valeurs à conserver.
    fill : str
        Valeur de remplacement.

    Returns
    -------
    pd.Series
        Série avec cardinalité réduite.
    """
    top_vals = series.value_counts().head(top_n).index
    return series.where(series.isin(top_vals), other=fill)


# ─────────────────────────────────────────────────────────────
# Fonction principale de prétraitement
# ─────────────────────────────────────────────────────────────
def preparer_donnees(chemin_csv: str, top_n_techniques: int = 15,
                     verbose: bool = True) -> tuple:
    """Charge le CSV et exécute la pipeline de prétraitement complète.

    Reproduit fidèlement les étapes validées sur la branche ``main`` :
    sélection des colonnes, nettoyage NaN, réduction de cardinalité,
    features temporelles, encodage de la cible.

    Parameters
    ----------
    chemin_csv : str
        Chemin vers le fichier ``echantillon.csv``.
    top_n_techniques : int, optional
        Nombre de techniques MITRE à conserver (défaut 15).
    verbose : bool, optional
        Afficher les étapes (défaut True).

    Returns
    -------
    tuple
        ``(X, y, df_model)`` où *X* est le DataFrame de features,
        *y* la Series cible encodée (0/1/2), et *df_model* le DataFrame
        complet avant séparation X/y.
    """
    # ── 1. Chargement ──
    echantillon = pd.read_csv(chemin_csv, low_memory=False)
    if verbose:
        print(f"Dimensions brutes : {echantillon.shape[0]:,} x {echantillon.shape[1]}")

    # ── 2. Sélection des colonnes ──
    df = echantillon[COLONNES_UTILES].copy()
    df = df.dropna(subset=['IncidentGrade'])
    if verbose:
        print(f"Colonnes conservées : {df.shape[1]}  |  "
              f"Lignes : {df.shape[0]:,}")

    # ── 3. Valeurs manquantes ──
    fill_unknown = ['MitreTechniques', 'LastVerdict', 'OSVersion']
    df[fill_unknown] = df[fill_unknown].fillna('Unknown')
    df['SuspicionLevel'] = df['SuspicionLevel'].fillna('None')
    for col in ['OSFamily', 'CountryCode', 'State', 'City']:
        df[col] = df[col].fillna('Unknown')
    assert df.isna().sum().sum() == 0, "Il reste des NaN !"
    if verbose:
        print("Aucune valeur manquante.")

    # ── 4. Simplification de MitreTechniques ──
    df['MitreTechnique_Main'] = df['MitreTechniques'].apply(
        extraire_technique_principale
    )
    top_techniques = (df['MitreTechnique_Main']
                      .value_counts()
                      .head(top_n_techniques)
                      .index)
    df['MitreTechnique_Main'] = df['MitreTechnique_Main'].where(
        df['MitreTechnique_Main'].isin(top_techniques), other='Other'
    )
    df = df.drop(columns=['MitreTechniques'])
    if verbose:
        print(f"Techniques MITRE distinctes : "
              f"{df['MitreTechnique_Main'].nunique()}")

    # ── 5. Réduction de cardinalité (géographiques + OS) ──
    df['City']        = reduire_cardinalite(df['City'], top_n=20)
    df['State']       = reduire_cardinalite(df['State'], top_n=20)
    df['CountryCode'] = reduire_cardinalite(df['CountryCode'], top_n=30)
    df['OSVersion']   = reduire_cardinalite(df['OSVersion'], top_n=15)

    # ── 6. Feature engineering temporel ──
    df_model = df.copy()
    if 'Timestamp' in df_model.columns:
        df_model['Timestamp'] = pd.to_datetime(
            df_model['Timestamp'], errors='coerce'
        )
        df_model['DayOfWeek'] = df_model['Timestamp'].dt.dayofweek
        df_model['Hour'] = df_model['Timestamp'].dt.hour
        df_model['IsWeekend'] = (df_model['DayOfWeek'] >= 5).astype(int)
        df_model['IsBusinessHour'] = (
            df_model['Hour'].between(8, 18).astype(int)
        )
        df_model.drop(columns=['Timestamp', 'DayOfWeek', 'Hour'],
                      errors='ignore', inplace=True)

    # ── 7. Encodage de la cible ──
    df_model['cible'] = df_model['IncidentGrade'].map(CIBLE_MAP)
    df_model = df_model.drop(columns=['IncidentGrade'])

    # Assurer le type str pour les catégorielles
    df_model['RegistryValueData'] = df_model['RegistryValueData'].astype(str)
    df_model['OSFamily'] = df_model['OSFamily'].astype(str)
    cat_cols = df_model.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()
    for col in cat_cols:
        df_model[col] = df_model[col].astype(str)

    # ── 8. Séparation X / y ──
    X = df_model.drop(columns=['cible'])
    y = df_model['cible']

    if verbose:
        print(f"\nPretraitement termine.")
        print(f"   Features : {X.shape[1]} colonnes")
        n = X.select_dtypes(exclude='object').columns.tolist()
        c = X.select_dtypes(include='object').columns.tolist()
        print(f"   Numériques    ({len(n)}) : {n}")
        print(f"   Catégorielles ({len(c)}) : {c}")
        print(f"\n   Distribution cible :")
        print(y.map(CIBLE_MAP_INV).value_counts().to_string())

    return X, y, df_model


# ─────────────────────────────────────────────────────────────
# Split + Preprocessing pipeline
# ─────────────────────────────────────────────────────────────
def split_et_transformer(X: pd.DataFrame, y: pd.Series,
                         test_size: float = 0.2,
                         random_state: int = 2026,
                         verbose: bool = True) -> tuple:
    """Split stratifié + ColumnTransformer (scale num, encode cat).

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Cible encodée.
    test_size : float
        Proportion du test set (défaut 0.2).
    random_state : int
        Graine aléatoire.
    verbose : bool
        Afficher les dimensions.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test, preprocessor,
        X_train_transformed, X_test_transformed, num_cols, cat_cols)``
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y
    )

    num_cols = X.select_dtypes(exclude='object').columns.tolist()
    cat_cols = X.select_dtypes(include='object').columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore',
                                  sparse_output=False), cat_cols),
        ]
    )

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    if verbose:
        print(f"Train : {X_train.shape[0]:,}  |  Test : {X_test.shape[0]:,}")
        print(f"Après transformation — "
              f"Train : {X_train_transformed.shape}  |  "
              f"Test : {X_test_transformed.shape}")

    return (X_train, X_test, y_train, y_test,
            preprocessor, X_train_transformed, X_test_transformed,
            num_cols, cat_cols)
