import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, precision_recall_curve, auc

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

def nettoyer(x):
    if pd.isnull(x):
        return "Unknown"
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


def ajouter_temps(df: pd.DataFrame) -> pd.DataFrame:
    df_time = df.copy()
    if 'Timestamp' in df_time.columns:
        ts = pd.to_datetime(df_time['Timestamp'], errors='coerce')
        df_time['Month'] = ts.dt.month.fillna(0).astype(int)
        df_time['IsWeekend'] = (ts.dt.dayofweek >= 5).fillna(0).astype(int)
        df_time['IsBusinessHour'] = ts.dt.hour.between(8, 18).fillna(0).astype(int)
    return df_time


def preparer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_prep = ajouter_temps(df)
    X = df_prep[COLONNES].fillna("Unknown")
    for col in COLONNES:
        X[col] = X[col].apply(nettoyer)
    return X


def predire(X: pd.DataFrame):
    X_pca = pca.transform(prep.transform(X))
    p_lr = lr.predict(X_pca)
    p_knn = knn.predict(X_pca)

    p_lr_proba = lr.predict_proba(X_pca)[:, 1] if hasattr(lr, "predict_proba") else None
    p_knn_proba = knn.predict_proba(X_pca)[:, 1] if hasattr(knn, "predict_proba") else None

    X_rf = rf_prep.transform(X)
    indices = rf_meta.get('selected_features_indices')
    if indices:
        X_rf = X_rf[:, indices]
    rf_proba = rf.predict_proba(X_rf)[:, 1]
    threshold = rf_meta.get('threshold', 0.5)
    p_rf = (rf_proba >= threshold).astype(int)

    return {
        "preds": {"LR": p_lr, "KNN": p_knn, "RF": p_rf},
        "probas": {"LR": p_lr_proba, "KNN": p_knn_proba, "RF": rf_proba},
        "threshold": threshold,
    }


def afficher_bilan_reference():
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


def afficher_definitions():
    with st.expander("Définitions rapides des métriques"):
        st.markdown(
            "- **Précision** : parmi les alertes signalées comme vraies, combien étaient réellement vraies.\n"
            "- **Rappel** : parmi toutes les vraies alertes, combien ont été détectées.\n"
            "- **F1-score** : moyenne entre précision et rappel pour équilibrer les deux."
        )


def appliquer_filtres(df: pd.DataFrame) -> pd.DataFrame:
    df_filtre = df.copy()
    with st.expander("Filtres rapides"):
        if "Month" in df_filtre.columns:
            mois_min, mois_max = int(df_filtre["Month"].min()), int(df_filtre["Month"].max())
            plage = st.slider("Mois", min_value=mois_min, max_value=mois_max, value=(mois_min, mois_max))
            df_filtre = df_filtre[(df_filtre["Month"] >= plage[0]) & (df_filtre["Month"] <= plage[1])]

        for col_name, label in [
            ("IsWeekend", "Week-end"),
            ("IsBusinessHour", "Heures ouvrées"),
            ("Category", "Catégorie"),
            ("CountryCode", "Pays"),
        ]:
            if col_name in df_filtre.columns:
                options = (
                    df_filtre[col_name]
                    .dropna()
                    .astype(str)
                    .value_counts()
                    .head(30)
                    .index
                    .tolist()
                )
                if options:
                    selection = st.multiselect(label, options, default=options)
                    df_filtre = df_filtre[df_filtre[col_name].astype(str).isin(selection)]
    return df_filtre


def afficher_kpis(preds, vrai=None):
    st.subheader("KPI rapides")
    if vrai is None:
        total = len(next(iter(preds.values())))
        st.metric("Alertes analysées", total)
        return

    total = len(vrai)
    taux_tp = float(np.mean(vrai)) if total else 0
    st.metric("Alertes analysées", total)
    st.metric("Taux TruePositive", f"{taux_tp:.1%}")

    cols = st.columns(3)
    for col, key, label in zip(cols, ["LR", "KNN", "RF"], ["Régression", "KNN", "Random Forest"]):
        col.metric("Modèle", label)
        col.metric("Precision", f"{precision_score(vrai, preds[key], zero_division=0):.2f}")
        col.metric("Recall", f"{recall_score(vrai, preds[key], zero_division=0):.2f}")
        col.metric("F1", f"{f1_score(vrai, preds[key], average='macro'):.2f}")

def afficher_graphiques(preds, vrai=None, probas=None):
    st.subheader("Vue rapide des prédictions")
    col1, col2, col3 = st.columns(3)
    for col, nom in zip([col1, col2, col3], ["LR", "KNN", "RF"]):
        counts = pd.Series(preds[nom]).value_counts().reindex([0, 1], fill_value=0)
        col.metric("Alertes TP (1)", int(counts[1]))
        col.metric("Alertes Non-TP (0)", int(counts[0]))

    counts_df = pd.DataFrame({
        "LR": pd.Series(preds["LR"]).value_counts(),
        "KNN": pd.Series(preds["KNN"]).value_counts(),
        "RF": pd.Series(preds["RF"]).value_counts(),
    }).fillna(0).astype(int)

    counts_long = counts_df.reset_index().melt(id_vars="index", var_name="Modèle", value_name="Nombre")
    counts_long["Classe"] = counts_long["index"].map({0: "Non-TP", 1: "TP"})
    fig_counts = px.bar(counts_long, x="Modèle", y="Nombre", color="Classe", barmode="group")
    st.plotly_chart(fig_counts, use_container_width=True)

    if vrai is not None:
        st.subheader("Matrice de confusion (résumé)")
        cols = st.columns(3)
        for col, nom in zip(cols, ["LR", "KNN", "RF"]):
            cm = pd.crosstab(vrai, preds[nom], rownames=['Vrai'], colnames=['Prédit'], dropna=False)
            col.write(f"**{nom}**")
            col.dataframe(cm)

    if vrai is not None and probas is not None:
        st.subheader("Courbes ROC & Precision-Recall")
        roc_fig = go.Figure()
        pr_fig = go.Figure()
        for key, nom in [("LR", "Régression"), ("KNN", "KNN"), ("RF", "Random Forest")]:
            proba = probas.get(key)
            if proba is None:
                continue
            fpr, tpr, _ = roc_curve(vrai, proba)
            roc_auc = auc(fpr, tpr)
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{nom} (AUC={roc_auc:.2f})"))

            precision, recall, _ = precision_recall_curve(vrai, proba)
            pr_auc = auc(recall, precision)
            pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"{nom} (AUC={pr_auc:.2f})"))

        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Hasard", line=dict(dash="dash")))
        roc_fig.update_layout(xaxis_title="FPR", yaxis_title="TPR", height=380)
        pr_fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=380)
        st.plotly_chart(roc_fig, use_container_width=True)
        st.plotly_chart(pr_fig, use_container_width=True)


st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Aller à",
    ["Vue d'ensemble", "Analyse par fichier", "Analyse manuelle"],
)

st.sidebar.markdown("---")
st.sidebar.write("**Modèles disponibles**")
st.sidebar.write("- Régression Logistique")
st.sidebar.write("- KNN")
st.sidebar.write("- Random Forest")

# Options prédéfinies pour le formulaire manuel
OPTIONS = {
    "AlertTitle": ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","Other"],
    "Category": ["InitialAccess","Exfiltration","SuspiciousActivity","CommandAndControl","Impact","CredentialAccess","Execution","Malware","Discovery","Persistence","DefenseEvasion","LateralMovement","Ransomware","UnwantedSoftware","Collection","PrivilegeEscalation","Exploit","CredentialStealing"],
    "MitreTechniques": ["Unknown","T1078","T1566.002","T1566","Other","T1110","T1133","T1566.001","T1046","T1087","T1190","T1559","T1568","T1003","T1027","T1071"],
    "ActionGrouped": ["Unknown","ContainAccount","IsolateDevice"],
    "ActionGranular": ["Unknown","Contain","Isolate"],
    "EntityType": ["Ip","User","MailMessage","Machine","Url","File","MailCluster","Process","Mailbox","RegistryValue","CloudApplication","DnsResolution","SecurityGroup","AzureResource","CloudLogonSession","SubmissionMail","OAuthApplication","RegistryKey","Other","IoTDevice","NicEntity","AccountSid","Unknown"],
    "ThreatFamily": ["Unknown","Other","Malgent","Emotet","Mimikatz","Webshell","Phish","Incommodious","Torpig","Suspicious","Exploit","NanoCore","Donoff","Mikatz","Bruteforce","IcedId","Codoso","CobaltStrike","Remcos","Petrwrap","Trickbot","Tiggre","BazarLoader","Qakbot","EvilProxy","Log4j","Cerber","EICAR_Test_File","AsyncRat","AgentTesla","Dridex"],
    "ResourceType": ["Unknown","Virtual Machine","Azure Arc machine","Other","App service","Key vault","Storage account","SQL server","DNS","Azure Kubernetes Service","Container registry","Cosmos DB"],
    "Roles": ["Unknown","Contextual","Destination","Suspicious","Source","Created","Related","Attacker","Affected"],
    "EvidenceRole": ["Related","Impacted"],
    "RegistryValueData": ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","Unknown"],
    "OSFamily": ["0","1","2","3","4","5"],
    "OSVersion": [str(i) for i in range(67)] + ["Unknown"],
    "AntispamDirection": ["Unknown","Inbound","Intraorg","Outbound"],
    "SuspicionLevel": ["Suspicious","Incriminated"],
    "LastVerdict": ["Unknown","Suspicious","Malicious","NoThreatsFound","Clean","Phishing","Spam"],
    "CountryCode": [str(i) for i in range(243)] + ["Other","Unknown"],
    "Month": list(range(1, 13)),
    "IsWeekend": [0, 1],
    "IsBusinessHour": [0, 1],
}

OS_FAMILY_NAMES = {
    "0": "Windows",
    "1": "Linux",
    "2": "macOS",
    "3": "iOS",
    "4": "Android",
    "5": "Autre / Inconnu",
}

OS_VERSION_NAMES = {
    "0":"Inconnu","1":"Windows 10 (1507)","2":"Windows 10 (1511)","3":"Windows 10 (1607)",
    "4":"Windows 10 (1703)","5":"Windows 10 (1709)","6":"Windows 10 (1803)","7":"Windows 10 (1809)",
    "8":"Windows 10 (1903)","9":"Windows 10 (1909)","10":"Windows 10 (2004)","11":"Windows 10 (20H2)",
    "12":"Windows 10 (21H1)","13":"Windows 10 (21H2)","14":"Windows 10 (22H2)","15":"Windows 11 (21H2)",
    "16":"Windows 11 (22H2)","17":"Windows 11 (23H2)","18":"Windows Server 2008","19":"Windows Server 2008 R2",
    "20":"Windows Server 2012","21":"Windows Server 2012 R2","22":"Windows Server 2016",
    "23":"Windows Server 2019","24":"Windows Server 2022","25":"Ubuntu 18.04","26":"Ubuntu 20.04",
    "27":"Ubuntu 22.04","28":"Ubuntu 24.04","29":"Debian 10","30":"Debian 11","31":"Debian 12",
    "32":"CentOS 7","33":"CentOS 8","34":"RHEL 7","35":"RHEL 8","36":"RHEL 9",
    "37":"Fedora 36","38":"Fedora 37","39":"Fedora 38","40":"SUSE 15","41":"openSUSE Leap",
    "42":"macOS 11 (Big Sur)","43":"macOS 12 (Monterey)","44":"macOS 13 (Ventura)",
    "45":"macOS 14 (Sonoma)","46":"macOS 15 (Sequoia)","47":"iOS 15","48":"iOS 16","49":"iOS 17",
    "50":"Android 11","51":"Android 12","52":"Android 13","53":"Android 14",
    "54":"Windows 7","55":"Windows 8","56":"Windows 8.1","57":"Windows XP",
    "58":"Alpine Linux","59":"Arch Linux","60":"Kali Linux","61":"Amazon Linux 2",
    "62":"Oracle Linux 8","63":"Rocky Linux 8","64":"Rocky Linux 9","65":"ChromeOS",
    "66":"Autre version","Unknown":"Inconnu",
}

COUNTRY_NAMES = {
    "0":"Afghanistan","1":"Albanie","2":"Algérie","3":"Andorre","4":"Angola","5":"Antigua-et-Barbuda",
    "6":"Argentine","7":"Arménie","8":"Australie","9":"Autriche","10":"Azerbaïdjan","11":"Bahamas",
    "12":"Bahreïn","13":"Bangladesh","14":"Barbade","15":"Biélorussie","16":"Belgique","17":"Belize",
    "18":"Bénin","19":"Bhoutan","20":"Bolivie","21":"Bosnie-Herzégovine","22":"Botswana","23":"Brésil",
    "24":"Brunei","25":"Bulgarie","26":"Burkina Faso","27":"Burundi","28":"Cambodge","29":"Cameroun",
    "30":"Canada","31":"Cap-Vert","32":"Centrafrique","33":"Tchad","34":"Chili","35":"Chine",
    "36":"Colombie","37":"Comores","38":"Congo","39":"Costa Rica","40":"Croatie","41":"Cuba",
    "42":"Chypre","43":"Tchéquie","44":"RD Congo","45":"Danemark","46":"Djibouti","47":"Dominique",
    "48":"République dominicaine","49":"Équateur","50":"Égypte","51":"Salvador","52":"Guinée équatoriale",
    "53":"Érythrée","54":"Estonie","55":"Eswatini","56":"Éthiopie","57":"Fidji","58":"Finlande",
    "59":"France","60":"Gabon","61":"Gambie","62":"Géorgie","63":"Allemagne","64":"Ghana","65":"Grèce",
    "66":"Grenade","67":"Guatemala","68":"Guinée","69":"Guinée-Bissau","70":"Guyana","71":"Haïti",
    "72":"Honduras","73":"Hongrie","74":"Islande","75":"Inde","76":"Indonésie","77":"Iran","78":"Irak",
    "79":"Irlande","80":"Israël","81":"Italie","82":"Jamaïque","83":"Japon","84":"Jordanie",
    "85":"Kazakhstan","86":"Kenya","87":"Kiribati","88":"Koweït","89":"Kirghizistan","90":"Laos",
    "91":"Lettonie","92":"Liban","93":"Lesotho","94":"Liberia","95":"Libye","96":"Liechtenstein",
    "97":"Lituanie","98":"Luxembourg","99":"Madagascar","100":"Malawi","101":"Malaisie","102":"Maldives",
    "103":"Mali","104":"Malte","105":"Îles Marshall","106":"Mauritanie","107":"Maurice","108":"Mexique",
    "109":"Micronésie","110":"Moldavie","111":"Monaco","112":"Mongolie","113":"Monténégro","114":"Maroc",
    "115":"Mozambique","116":"Myanmar","117":"Namibie","118":"Nauru","119":"Népal","120":"Pays-Bas",
    "121":"Nouvelle-Zélande","122":"Nicaragua","123":"Niger","124":"Nigeria","125":"Corée du Nord",
    "126":"Macédoine du Nord","127":"Norvège","128":"Oman","129":"Pakistan","130":"Palaos","131":"Panama",
    "132":"Papouasie-Nouvelle-Guinée","133":"Paraguay","134":"Pérou","135":"Philippines","136":"Pologne",
    "137":"Portugal","138":"Qatar","139":"Roumanie","140":"Russie","141":"Rwanda","142":"Saint-Kitts-et-Nevis",
    "143":"Sainte-Lucie","144":"Saint-Vincent","145":"Samoa","146":"Saint-Marin","147":"Sao Tomé-et-Príncipe",
    "148":"Arabie saoudite","149":"Sénégal","150":"Serbie","151":"Seychelles","152":"Sierra Leone",
    "153":"Singapour","154":"Slovaquie","155":"Slovénie","156":"Îles Salomon","157":"Somalie",
    "158":"Afrique du Sud","159":"Corée du Sud","160":"Soudan du Sud","161":"Espagne","162":"Sri Lanka",
    "163":"Soudan","164":"Suriname","165":"Suède","166":"Suisse","167":"Syrie","168":"Taïwan",
    "169":"Tadjikistan","170":"Tanzanie","171":"Thaïlande","172":"Timor oriental","173":"Togo",
    "174":"Tonga","175":"Trinité-et-Tobago","176":"Tunisie","177":"Turquie","178":"Turkménistan",
    "179":"Tuvalu","180":"Ouganda","181":"Ukraine","182":"Émirats arabes unis","183":"Royaume-Uni",
    "184":"États-Unis","185":"Uruguay","186":"Ouzbékistan","187":"Vanuatu","188":"Vatican",
    "189":"Venezuela","190":"Viêt Nam","191":"Yémen","192":"Zambie","193":"Zimbabwe",
    "194":"Kosovo","195":"Palestine","196":"Hong Kong","197":"Macao","198":"Porto Rico",
    "199":"Guam","200":"Îles Vierges américaines","201":"Samoa américaines","202":"Bermudes",
    "203":"Îles Caïmans","204":"Curaçao","205":"Aruba","206":"Sint Maarten","207":"Turks-et-Caïcos",
    "208":"Îles Vierges britanniques","209":"Anguilla","210":"Montserrat","211":"Guadeloupe",
    "212":"Martinique","213":"Guyane française","214":"Réunion","215":"Mayotte","216":"Nouvelle-Calédonie",
    "217":"Polynésie française","218":"Wallis-et-Futuna","219":"Saint-Pierre-et-Miquelon",
    "220":"Saint-Barthélemy","221":"Saint-Martin","222":"Gibraltar","223":"Îles Féroé","224":"Groenland",
    "225":"Jersey","226":"Guernesey","227":"Île de Man","228":"Åland","229":"Svalbard",
    "230":"Sahara occidental","231":"Somaliland","232":"Transnistrie","233":"Abkhazie","234":"Ossétie du Sud",
    "235":"Chypre du Nord","236":"Haut-Karabakh","237":"Taïwan (alt)","238":"Antarctique",
    "239":"Île Christmas","240":"Îles Cocos","241":"Île Norfolk","242":"Tokelau",
    "Other":"Autre","Unknown":"Inconnu",
}

if page == "Vue d'ensemble":
    afficher_bilan_reference()
    afficher_definitions()
    st.info(
        "Utilisez les sections ‘Analyse par fichier’ ou ‘Analyse manuelle’ pour tester des alertes."
    )

elif page == "Analyse par fichier":
    st.write("### Analyse de fichiers CSV")
    uploaded_file = st.file_uploader("Charger un fichier CSV", type="csv")

    if st.button("Utiliser un échantillon aléatoire (2000 lignes)"):
        try:
            df_full = pd.read_csv("test_user_interface.csv")
            st.session_state['df_test'] = df_full.sample(n=min(2000, len(df_full)), random_state=2026)
        except Exception as e:
            st.error(f"Erreur lors du chargement de test_user_interface.csv : {e}")

    df = None
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    elif 'df_test' in st.session_state:
        df = st.session_state['df_test']

    if df is not None:
        df = ajouter_temps(df)
        df = appliquer_filtres(df)
        if df.empty:
            st.warning("Aucune ligne après filtrage. Ajustez les filtres.")
        else:
            X = preparer_features(df)
            with st.spinner("Analyse en cours..."):
                resultats = predire(X)
            st.success(f"Analyse effectuée sur {len(df)} lignes.")

            preds = resultats["preds"]
            probas = resultats["probas"]
            vrai = None
            if 'IncidentGrade' in df.columns:
                vrai = df['IncidentGrade'].apply(
                    lambda x: 1 if str(x).lower().strip() in ['truepositive', '1'] else 0
                )

                bilan = []
                for nom, key in [("Regression Logistique", "LR"), ("KNN", "KNN"), ("Random Forest", "RF")]:
                    bilan.append({
                        "Modele": nom,
                        "Precision": precision_score(vrai, preds[key], zero_division=0),
                        "Rappel": recall_score(vrai, preds[key], zero_division=0),
                        "F1-Macro": f1_score(vrai, preds[key], average='macro')
                    })
                st.subheader("Performance sur le fichier")
                st.table(pd.DataFrame(bilan))
            else:
                st.info("Résultats des prédictions (0 = Benign/False, 1 = TruePositive)")
                st.dataframe(pd.DataFrame({"LR": preds["LR"], "KNN": preds["KNN"], "RF": preds["RF"]}))

            afficher_kpis(preds, vrai)
            afficher_graphiques(preds, vrai, probas)

            if "Month" in df.columns:
                df_month = df.copy()
                df_month["Prediction_RF"] = preds["RF"]
                fig_month = px.histogram(
                    df_month,
                    x="Month",
                    color="Prediction_RF",
                    barmode="group",
                    title="Répartition des prédictions RF par mois",
                )
                st.plotly_chart(fig_month, use_container_width=True)

            with st.expander("Voir les détails par ligne"):
                df_final = df.copy()
                df_final.insert(0, "Pred_RF", preds["RF"])
                df_final.insert(0, "Pred_KNN", preds["KNN"])
                df_final.insert(0, "Pred_LR", preds["LR"])
                if probas["LR"] is not None:
                    df_final.insert(3, "Proba_LR", probas["LR"])
                if probas["KNN"] is not None:
                    df_final.insert(4, "Proba_KNN", probas["KNN"])
                df_final.insert(5, "Proba_RF", probas["RF"])
                st.dataframe(df_final)

            csv = df_final.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger les résultats (CSV)",
                csv,
                file_name="predictions_incidents.csv",
                mime="text/csv",
            )

elif page == "Analyse manuelle":
    st.write("### Analyse d'une alerte (saisie manuelle)")

    with st.form("formulaire_alerte"):
        col1, col2 = st.columns(2)

        data = {}
        # Champs spéciaux IsWeekend / IsBusinessHour avec contrainte
        weekend_labels = {0: "Non (jour de semaine)", 1: "Oui (week-end)"}
        bh_labels = {0: "Non", 1: "Oui (8h-18h)"}

        for idx, col_name in enumerate(COLONNES):
            container = col1 if idx % 2 == 0 else col2
            opts = OPTIONS.get(col_name, ["Unknown"])

            if col_name == "IsWeekend":
                data[col_name] = container.selectbox(
                    "Week-end ?", [0, 1], index=0,
                    format_func=lambda x: weekend_labels[x]
                )
            elif col_name == "IsBusinessHour":
                if data.get("IsWeekend") == 1:
                    container.selectbox(
                        "Heure d'affaire ?", [0], index=0,
                        format_func=lambda x: "Non (week-end)"
                    )
                    data[col_name] = 0
                else:
                    data[col_name] = container.selectbox(
                        "Heure d'affaire ?", [0, 1], index=0,
                        format_func=lambda x: bh_labels[x]
                    )
            elif col_name == "CountryCode":
                data[col_name] = container.selectbox(
                    col_name, opts, index=0,
                    format_func=lambda x: COUNTRY_NAMES.get(str(x), str(x))
                )
            elif col_name == "OSFamily":
                data[col_name] = container.selectbox(
                    col_name, opts, index=0,
                    format_func=lambda x: OS_FAMILY_NAMES.get(str(x), str(x))
                )
            elif col_name == "OSVersion":
                data[col_name] = container.selectbox(
                    col_name, opts, index=0,
                    format_func=lambda x: OS_VERSION_NAMES.get(str(x), str(x))
                )
            elif col_name == "Month":
                month_labels = {1:"Janvier",2:"Février",3:"Mars",4:"Avril",5:"Mai",6:"Juin",
                                7:"Juillet",8:"Août",9:"Septembre",10:"Octobre",11:"Novembre",12:"Décembre"}
                data[col_name] = container.selectbox(
                    "Mois", opts, index=0,
                    format_func=lambda x: month_labels.get(x, str(x))
                )
            else:
                data[col_name] = container.selectbox(col_name, opts, index=0)

        submitted = st.form_submit_button("Analyser cette alerte")

    if submitted:
        df_case = pd.DataFrame([data])
        X = preparer_features(df_case)
        resultats = predire(X)
        preds = resultats["preds"]
        probas = resultats["probas"]

        st.subheader("Résultat de la prédiction")
        labels = {
            "LR": "TruePositive" if int(preds["LR"][0]) == 1 else "Non-TruePositive",
            "KNN": "TruePositive" if int(preds["KNN"][0]) == 1 else "Non-TruePositive",
            "RF": "TruePositive" if int(preds["RF"][0]) == 1 else "Non-TruePositive",
        }

        st.table(pd.DataFrame({
            "Modèle": ["Régression Logistique", "KNN", "Random Forest"],
            "Prédiction": [labels["LR"], labels["KNN"], labels["RF"]],
            "Probabilité (TP)": [
                probas["LR"][0] if probas["LR"] is not None else None,
                probas["KNN"][0] if probas["KNN"] is not None else None,
                probas["RF"][0],
            ],
        }))

        probas_plot = {
            "Régression": probas["LR"][0] if probas["LR"] is not None else None,
            "KNN": probas["KNN"][0] if probas["KNN"] is not None else None,
            "Random Forest": probas["RF"][0],
        }
        probas_df = (
            pd.DataFrame({"Modèle": list(probas_plot.keys()), "Probabilité": list(probas_plot.values())})
            .dropna()
        )
        if not probas_df.empty:
            fig_proba = px.bar(probas_df, x="Modèle", y="Probabilité", range_y=[0, 1])
            st.plotly_chart(fig_proba, use_container_width=True)

        st.caption(
            f"Seuil Random Forest : {resultats['threshold']:.2f} (probabilité >= seuil = TruePositive)."
        )
