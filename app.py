import streamlit as st
import pandas as pd
import joblib
import json
from sklearn.metrics import f1_score, precision_score, recall_score

st.set_page_config(page_title="Classification Incidents", layout="wide")
st.title("Classification des Incidents de Sécurité")

# Chargement des fichiers
@st.cache_resource
def charger_modeles():
    prep = joblib.load('models/preprocessor.joblib')
    rf_prep = joblib.load('models/rf_preprocessor.joblib')
    pca = joblib.load('models/pca.joblib')
    lr = joblib.load('models/best_regression.joblib')
    knn = joblib.load('models/best_knn.joblib')
    rf = joblib.load('models/rf_model.joblib')
    lr_meta = json.load(open('models/reg_metadata.json'))
    knn_meta = json.load(open('models/knn_metadata.json'))
    rf_meta = json.load(open('models/rf_metadata.json'))
    return prep, rf_prep, pca, lr, knn, rf, lr_meta, knn_meta, rf_meta

prep, rf_prep, pca, lr, knn, rf, lr_meta, knn_meta, rf_meta = charger_modeles()

# --- SECTION 1 : RÉFÉRENCE ---
st.write("### Performances de référence (Entrainement)")
bilan_ref = []
for nom, meta in [("Regression Logistique", lr_meta), ("KNN", knn_meta), ("Random Forest", rf_meta)]:
    m = meta['training_metrics']
    bilan_ref.append({
        "Modele": nom,
        "Precision": f"{m['precision']:.4f}",
        "Rappel": f"{m['recall']:.4f}",
        "F1-Macro": f"{m['f1_macro']:.4f}"
    })
st.table(pd.DataFrame(bilan_ref))

st.markdown("---")

def nettoyer(x):
    if pd.isnull(x): return "Unknown"
    if isinstance(x, (int, float)) and float(x).is_integer():
        return str(int(x))
    return str(x)

# Les 20 colonnes requises
COLONNES = [
    'AlertTitle', 'Category', 'MitreTechniques', 'ActionGrouped', 'ActionGranular', 
    'EntityType', 'ThreatFamily', 'ResourceType', 'Roles', 'EvidenceRole', 
    'RegistryValueData', 'OSFamily', 'OSVersion', 'AntispamDirection', 
    'SuspicionLevel', 'LastVerdict', 'CountryCode', 'Month', 'IsWeekend', 'IsBusinessHour'
]

uploaded_file = st.file_uploader("Charger un fichier CSV (ex: test_upload.csv)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Extraction du temps si absent
    if 'Timestamp' in df.columns:
        ts = pd.to_datetime(df['Timestamp'], errors='coerce')
        df['Month'] = ts.dt.month.fillna(0).astype(int)
        df['IsWeekend'] = (ts.dt.dayofweek >= 5).fillna(0).astype(int)
        df['IsBusinessHour'] = ts.dt.hour.between(8, 18).fillna(0).astype(int)

    # Preparation des donnees
    X = df[COLONNES].fillna("Unknown")
    for col in COLONNES:
        X[col] = X[col].apply(nettoyer)
    
    with st.spinner("Analyse en cours..."):
        # Predictions
        X_pca = pca.transform(prep.transform(X))
        p_lr = lr.predict(X_pca)
        p_knn = knn.predict(X_pca)
        
        X_rf = rf_prep.transform(X)
        indices = rf_meta.get('selected_features_indices')
        if indices: X_rf = X_rf[:, indices]
        p_rf = (rf.predict_proba(X_rf)[:, 1] >= rf_meta.get('threshold', 0.5)).astype(int)

        # Affichage des resultats
        st.write(f"Analyse effectuee sur {len(df)} lignes.")
        
        if 'IncidentGrade' in df.columns:
            vrai = df['IncidentGrade'].apply(lambda x: 1 if str(x).lower().strip() in ['truepositive', '1'] else 0)
            
            bilan = []
            for nom, preds in [("Regression Logistique", p_lr), ("KNN", p_knn), ("Random Forest", p_rf)]:
                bilan.append({
                    "Modele": nom,
                    "Precision": precision_score(vrai, preds, zero_division=0),
                    "Rappel": recall_score(vrai, preds, zero_division=0),
                    "F1-Macro": f1_score(vrai, preds, average='macro')
                })
            st.table(pd.DataFrame(bilan))
        else:
            st.info("Resultats des predictions (0 = Benign/False, 1 = TruePositive)")
            st.dataframe(pd.DataFrame({"LR": p_lr, "KNN": p_knn, "RF": p_rf}))

        # Tableau detaile
        with st.expander("Voir les details par ligne"):
            df_final = df.copy()
            df_final.insert(0, "Pred_RF", p_rf)
            df_final.insert(0, "Pred_KNN", p_knn)
            df_final.insert(0, "Pred_LR", p_lr)
            st.dataframe(df_final)
