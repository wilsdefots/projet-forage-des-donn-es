import streamlit as st
import pandas as pd
import joblib
import json
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix

# CONFIGURATION ET CHARGEMENT
st.set_page_config(page_title="Comparaison KNN vs RF", layout="wide")
st.title("Comparaison des Modèles de Sécurité")

# Chemins des modèles
models_path = {
    'KNN': ['models/best_knn.joblib', 'models/preprocessor.joblib', 'models/pca.joblib'],
    'RF': ['models/rf_model.joblib', 'models/rf_preprocessor.joblib', 'models/rf_metadata.json']
}

def clean_data(df):
    """Prépare les colonnes en forçant les types pour éviter les erreurs de comparaison."""
    num_cols = ['RegistryValueData', 'OSFamily', 'OSVersion', 'Month', 'IsWeekend', 'IsBusinessHour']
    cat_cols = ['AlertTitle', 'Category', 'MitreTechniques', 'ActionGrouped', 'ActionGranular', 
                'EntityType', 'ThreatFamily', 'ResourceType', 'Roles', 'EvidenceRole', 
                'AntispamDirection', 'SuspicionLevel', 'LastVerdict', 'CountryCode']
    
    df = df.copy()
    # Gestion du temps (nécessaire car le modèle attend Month, IsWeekend et IsBusinessHour)
    if 'Timestamp' in df.columns:
        ts = pd.to_datetime(df['Timestamp'], errors='coerce')
        df['Month'], df['IsWeekend'] = ts.dt.month, (ts.dt.dayofweek >= 5).fillna(0).astype(int)
        df['IsBusinessHour'] = ts.dt.hour.between(8, 18).fillna(0).astype(int)

    # Alignement des types
    for c in num_cols:
        df[c] = pd.to_numeric(df.get(c, 0), errors='coerce').fillna(0).astype(int)
            
    for c in cat_cols:
        df[c] = df.get(c, "Unknown").fillna("Unknown").astype(str)
            
    return df[num_cols + cat_cols]

# INTERFACE PRINCIPALE
uploaded_file = st.file_uploader("Charger un fichier brut (IncidentGrade) pour comparaison", type="csv")

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    X_clean = clean_data(df_raw)
    
    # Cible réelle (IncidentGrade)
    y_true = None
    if 'IncidentGrade' in df_raw.columns:
        y_true = df_raw['IncidentGrade'].apply(lambda x: 1 if str(x).lower().strip() in ['truepositive', '1'] else 0)

    results = pd.DataFrame({'Alerte': df_raw['AlertTitle']})
    if y_true is not None: results['Vraie Cible'] = y_true.map({0: "Non-TP", 1: "TruePositive"})

    metrics_list = []

    # EXECUTION KNN
    if all(os.path.exists(p) for p in models_path['KNN']):
        model = joblib.load(models_path['KNN'][0])
        prep = joblib.load(models_path['KNN'][1])
        pca = joblib.load(models_path['KNN'][2])
        
        X_knn = pca.transform(prep.transform(X_clean))
        preds = model.predict(X_knn)
        results['Pred (KNN)'] = pd.Series(preds).map({0: "Non-TP", 1: "TruePositive"})
        
        if y_true is not None:
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            metrics_list.append({
                'name': 'KNN',
                'acc': accuracy_score(y_true, preds),
                'prec': precision_score(y_true, preds, zero_division=0),
                'rec': recall_score(y_true, preds, zero_division=0),
                'f1': f1_score(y_true, preds, average='macro'),
                'mcc': matthews_corrcoef(y_true, preds),
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
            })

    # --- EXECUTION RANDOM FOREST ---
    if all(os.path.exists(p) for p in models_path['RF']):
        model = joblib.load(models_path['RF'][0])
        prep = joblib.load(models_path['RF'][1])
        meta = json.load(open(models_path['RF'][2]))
        
        X_rf = prep.transform(X_clean)
        if 'selected_features_indices' in meta:
            X_rf = X_rf[:, meta['selected_features_indices']]
            
        probs = model.predict_proba(X_rf)[:, 1]
        preds = (probs >= meta.get('threshold', 0.5)).astype(int)
        results['Pred (RF)'] = pd.Series(preds).map({0: "Non-TP", 1: "TruePositive"})
        
        if y_true is not None:
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            metrics_list.append({
                'name': 'Random Forest',
                'acc': accuracy_score(y_true, preds),
                'prec': precision_score(y_true, preds, zero_division=0),
                'rec': recall_score(y_true, preds, zero_division=0),
                'f1': f1_score(y_true, preds, average='macro'),
                'mcc': matthews_corrcoef(y_true, preds),
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
            })

    # RESULTATS
    st.subheader("Résultats de la Comparaison")
    st.dataframe(results, use_container_width=True)

    if metrics_list:
        st.subheader("Comparaison des Performances")
        # Création d'un tableau comparatif propre
        comparison_df = pd.DataFrame(metrics_list).set_index('name')
        
        # Sélection et renommage des colonnes pour l'affichage
        display_df = comparison_df[['acc', 'prec', 'rec', 'f1', 'mcc', 'tp', 'fp', 'tn', 'fn']].T
        display_df.index = [
            'Accuracy', 'Précision', 'Rappel (Recall)', 'Macro F1-Score', 
            'Matthews Correlation (MCC)', 'Vrais Positifs (TP)', 'Faux Positifs (FP)', 
            'Vrais Négatifs (TN)', 'Faux Négatifs (FN)'
        ]
        
        # Affichage du tableau statique
        st.table(display_df)
else:
    st.info("Veuillez uploader un fichier pour lancer les prédictions.")
