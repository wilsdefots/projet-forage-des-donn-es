# -- Bibliotheques standard --
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json
import joblib
warnings.filterwarnings('ignore')

# -- Scikit-learn : modelisation --
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    GridSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier 

# -- Scikit-learn : metriques --
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score, f1_score,
    precision_score, recall_score
)

# -- Constantes --
CIBLE_MAP = {'Non-TruePositive': 0, 'TruePositive': 1}
CIBLE_MAP_INV = {v: k for k, v in CIBLE_MAP.items()}
CIBLE_MAP_RF = {'Non-TruePositive': 0, 'TruePositive': 1}
TARGET_NAMES = ['Non-TP (0)', 'TP (1)']
print("Bibliotheques importees avec succes.")

#%%
# Prétraitement des données pour KNN et régression
data = np.load('donnees_transformees.npz', allow_pickle=True)
X_train_transformed = data['X_train']
X_test_transformed = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
feature_names_all = data['feature_names'].tolist()
# -- 3.1 Analyse de la variance --

# Fit PCA complet pour analyser la distribution de la variance
pca_full = PCA(random_state=2026)
pca_full.fit(X_train_transformed)

variance_cumulee = np.cumsum(pca_full.explained_variance_ratio_)

# Nombre de composantes pour 95% de variance
n_composantes_95 = np.argmax(variance_cumulee >= 0.95) + 1


# -- 3.3 Application de la PCA (95% de variance) --

pca = PCA(n_components=0.95, random_state=2026)
X_train_pca = pca.fit_transform(X_train_transformed)
X_test_pca  = pca.transform(X_test_transformed)

#%%

# -- Régression logistique : 4.1 -- Sans PCA vs Avec PCA --

# --- Sans PCA (toutes les features) ---
logistic_baseline_nopca = LogisticRegression(
    max_iter=1000,
    random_state=42
)
logistic_baseline_nopca.fit(X_train_transformed, y_train)
y_pred_bl_nopca = logistic_baseline_nopca.predict(X_test_transformed)

acc_bl_nopca = accuracy_score(y_test, y_pred_bl_nopca)
f1_bl_nopca  = f1_score(y_test, y_pred_bl_nopca, average='macro')

# --- Avec PCA ---
logistic_baseline = LogisticRegression(
    max_iter=1000,
    random_state=42
)
logistic_baseline.fit(X_train_pca, y_train)
y_pred_baseline = logistic_baseline.predict(X_test_pca)

acc_baseline = accuracy_score(y_test, y_pred_baseline)
f1_baseline  = f1_score(y_test, y_pred_baseline, average='macro')

#%%
# -- Régression : Utilisation du RandomizedSearch --
# GridSearch prenait trop de temps à run, donc RandomizedSearch est utilisé à la place.
param_distributions = {
    'C': np.logspace(-3, 3, 20),
    'penalty': ['l1', 'l2'],
    'solver': ['saga'],
    'class_weight': [None, 'balanced']
}

log_reg = LogisticRegression(max_iter=1000, random_state=42)


grid_search = RandomizedSearchCV (
    estimator=log_reg,
    param_distributions=param_distributions,
    scoring='f1_macro',
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=2026),
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid_search.fit(X_train_pca, y_train)

n_iter = 20
n_splits = 3
n_fits = n_iter * n_splits

# -- 6.2 Resultats detailles du RandomizedSearch (logistic) --

results_df = pd.DataFrame(grid_search.cv_results_)

# Adjust columns for logistic regression
cols_display = [
    'param_C', 'param_penalty', 'param_class_weight',
    'mean_test_score', 'std_test_score', 'mean_train_score', 'rank_test_score'
]

results_display = (
    results_df[cols_display]
    .sort_values('rank_test_score')
    .head(15)
    .reset_index(drop=True)
)

results_display.columns = [
    'C', 'Penalty', 'Class Weight',
    'F1 Test (CV)', 'Std Test', 'F1 Train (CV)', 'Rang'
]

# -- 7.1 Predictions avec le meilleur modele --

best_regression = grid_search.best_estimator_
y_pred_best = best_regression.predict(X_test_pca)

# Metriques detaillees
reg_acc  = accuracy_score(y_test, y_pred_best)
reg_f1   = f1_score(y_test, y_pred_best, average='macro')
reg_prec = precision_score(y_test, y_pred_best, average='macro')
reg_rec  = recall_score(y_test, y_pred_best, average='macro')

print("=" * 65)
print("     EVALUATION FINALE -- RÉGRESSION OPTIMISE + PCA (sur jeu de test)")
print("=" * 65)
print(f"\n  Hyperparametres optimaux :")
for param, val in grid_search.best_params_.items():
    print(f"    {param} = {val}")
print(f"  PCA : {X_train_pca.shape[1]} composantes ({pca.explained_variance_ratio_.sum()*100:.1f}% variance)")
print(f"\n  Accuracy        : {reg_acc:.4f}")
print(f"  Macro Precision : {reg_prec:.4f}")
print(f"  Macro Recall    : {reg_rec:.4f}")
print(f"  Macro F1-Score  : {reg_f1:.4f}")
print(f"\n{'-' * 65}")
print("\nRapport de classification detaille :")
print(classification_report(
    y_test, y_pred_best,
    target_names=['Non-TruePositive', 'TruePositive']
))

# -- 7.3 Comparaison : Sans PCA / PCA Baseline / PCA Optimise --

comparison = pd.DataFrame({
    'Metrique': ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-Score'],
    'Sans PCA': [
        acc_bl_nopca,
        precision_score(y_test, y_pred_bl_nopca, average='macro'),
        recall_score(y_test, y_pred_bl_nopca, average='macro'),
        f1_bl_nopca
    ],
    'PCA + Baseline': [
        acc_baseline,
        precision_score(y_test, y_pred_baseline, average='macro'),
        recall_score(y_test, y_pred_baseline, average='macro'),
        f1_baseline
    ],
    'PCA + Optimise': [reg_acc, reg_prec, reg_rec, reg_f1]
})
comparison['Gain total'] = comparison['PCA + Optimise'] - comparison['Sans PCA']
comparison['Gain (%)'] = (comparison['Gain total'] / comparison['Sans PCA'] * 100).round(2)

print("=" * 90)
print("     COMPARAISON : SANS PCA  vs  PCA + BASELINE  vs  PCA + OPTIMISE")
print("=" * 90)
print(comparison.to_string(index=False))

#%%
# -- Régression : 9.1 Analyse des erreurs par classe --

cm = confusion_matrix(y_test, y_pred_best)
classes = ['Non-TruePositive', 'TruePositive']

print("Analyse detaillee des erreurs par classe :\n")
for i, cls in enumerate(classes):
    total    = cm[i].sum()
    correct  = cm[i][i]
    errors   = total - correct
    error_rate = errors / total * 100
    
    print(f"  {cls} (n={total})")
    print(f"     Bien classes : {correct} ({correct/total*100:.1f}%)")
    print(f"     Mal classes  : {errors} ({error_rate:.1f}%)")
    
    # Detail des confusions
    for j, other_cls in enumerate(classes):
        if i != j and cm[i][j] > 0:
            print(f"       Confondu avec {other_cls} : {cm[i][j]} ({cm[i][j]/total*100:.1f}%)")
    print()

# Taux d'erreur global
total_errors = (y_test != y_pred_best).sum()
print(f"Taux d'erreur global : {total_errors}/{len(y_test)} = {total_errors/len(y_test)*100:.2f}%")
#%%

# -- 4.1 KNN Baseline (K=5) -- Sans PCA vs Avec PCA --

# --- Sans PCA (toutes les features) ---
knn_baseline_nopca = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2, n_jobs=-1)
knn_baseline_nopca.fit(X_train_transformed, y_train)
y_pred_bl_nopca = knn_baseline_nopca.predict(X_test_transformed)

acc_bl_nopca = accuracy_score(y_test, y_pred_bl_nopca)
f1_bl_nopca  = f1_score(y_test, y_pred_bl_nopca, average='macro')

# --- Avec PCA ---
knn_baseline = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2, n_jobs=-1)
knn_baseline.fit(X_train_pca, y_train)
y_pred_baseline = knn_baseline.predict(X_test_pca)

acc_baseline = accuracy_score(y_test, y_pred_baseline)
f1_baseline  = f1_score(y_test, y_pred_baseline, average='macro')

# -- 5.1 Elbow Method : K de 1 a 21 (impairs) --
# 3-fold CV sur donnees PCA : ~25k observations par fold = estimation stable

k_range = list(range(1, 22, 2))  # [1, 3, 5, 7, ..., 21]
cv_scores_mean = []
cv_scores_std  = []

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    scores = cross_val_score(knn, X_train_pca, y_train, cv=skf,
                             scoring='f1_macro', n_jobs=-1)
    cv_scores_mean.append(scores.mean())
    cv_scores_std.append(scores.std())
    

best_idx = np.argmax(cv_scores_mean)
best_k   = k_range[best_idx]


# -- 6.1 GridSearchCV -- Optimisation des hyperparametres --
# 3-fold CV sur donnees PCA -> rapide
# Ref. : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

# Grille centree autour du K optimal
k_fine_range = list(range(max(1, best_k - 4), best_k + 6, 2))

param_grid = {
    'n_neighbors': k_fine_range,
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski'],
    'p': [1, 2]  # p=1 : Manhattan, p=2 : Euclidienne
}

grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(n_jobs=-1),
    param_grid=param_grid,
    scoring='f1_macro',
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=2026),
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)
grid_search.fit(X_train_pca, y_train)


# -- 6.2 Resultats detailles du GridSearchCV --

results_df = pd.DataFrame(grid_search.cv_results_)


# Verification du surapprentissage
best_train = results_df.loc[results_df['rank_test_score'] == 1, 'mean_train_score'].values[0]
best_test  = grid_search.best_score_
gap = best_train - best_test
print(f"\nEcart Train-Test (meilleur modele) : {gap:.4f}")
if gap > 0.05:
    print("   Ecart significatif -> risque de surapprentissage")
else:
    print("   Ecart acceptable -> pas de surapprentissage majeur")

    # -- 5.3 Visualisation des resultats du GridSearch --

# --- Graphique 2 : Train vs Test score (meilleure config) ---
best_w = grid_search.best_params_['weights']
best_p = grid_search.best_params_['p']
mask_best = (results_df['param_weights'] == best_w) & (results_df['param_p'] == best_p)
subset_best = results_df[mask_best].sort_values('param_n_neighbors')

# -- 7.1 Predictions avec le meilleur modele --

best_knn = grid_search.best_estimator_
y_pred_best = best_knn.predict(X_test_pca)

# Metriques detaillees
acc_best  = accuracy_score(y_test, y_pred_best)
f1_best   = f1_score(y_test, y_pred_best, average='macro')
prec_best = precision_score(y_test, y_pred_best, average='macro')
rec_best  = recall_score(y_test, y_pred_best, average='macro')

print("=" * 65)
print("     EVALUATION FINALE -- KNN OPTIMISE + PCA (sur jeu de test)")
print("=" * 65)
print(f"\n  Hyperparametres optimaux :")
for param, val in grid_search.best_params_.items():
    print(f"    {param} = {val}")
print(f"  PCA : {X_train_pca.shape[1]} composantes ({pca.explained_variance_ratio_.sum()*100:.1f}% variance)")
print(f"\n  Accuracy        : {acc_best:.4f}")
print(f"  Macro Precision : {prec_best:.4f}")
print(f"  Macro Recall    : {rec_best:.4f}")
print(f"  Macro F1-Score  : {f1_best:.4f}")
print(f"\n{'-' * 65}")
print("\nRapport de classification detaille :")
print(classification_report(
    y_test, y_pred_best,
    target_names=['Non-TruePositive', 'TruePositive']
))


# -- 7.3 Comparaison : Sans PCA / PCA Baseline / PCA Optimise --

comparison = pd.DataFrame({
    'Metrique': ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-Score'],
    'Sans PCA (K=5)': [
        acc_bl_nopca,
        precision_score(y_test, y_pred_bl_nopca, average='macro'),
        recall_score(y_test, y_pred_bl_nopca, average='macro'),
        f1_bl_nopca
    ],
    'PCA + Baseline (K=5)': [
        acc_baseline,
        precision_score(y_test, y_pred_baseline, average='macro'),
        recall_score(y_test, y_pred_baseline, average='macro'),
        f1_baseline
    ],
    'PCA + Optimise': [acc_best, prec_best, rec_best, f1_best]
})
comparison['Gain total'] = comparison['PCA + Optimise'] - comparison['Sans PCA (K=5)']
comparison['Gain (%)'] = (comparison['Gain total'] / comparison['Sans PCA (K=5)'] * 100).round(2)

print("=" * 90)
print("     COMPARAISON : SANS PCA  vs  PCA + BASELINE  vs  PCA + OPTIMISE")
print("=" * 90)
print(comparison.to_string(index=False))


# -- 9.1 Analyse des erreurs par classe --

cm = confusion_matrix(y_test, y_pred_best)
classes = ['Non-TruePositive', 'TruePositive']

print("Analyse detaillee des erreurs par classe :\n")
for i, cls in enumerate(classes):
    total    = cm[i].sum()
    correct  = cm[i][i]
    errors   = total - correct
    error_rate = errors / total * 100
    
    print(f"  {cls} (n={total})")
    print(f"     Bien classes : {correct} ({correct/total*100:.1f}%)")
    print(f"     Mal classes  : {errors} ({error_rate:.1f}%)")
    
    # Detail des confusions
    for j, other_cls in enumerate(classes):
        if i != j and cm[i][j] > 0:
            print(f"       Confondu avec {other_cls} : {cm[i][j]} ({cm[i][j]/total*100:.1f}%)")
    print()

# Taux d'erreur global
total_errors = (y_test != y_pred_best).sum()
print(f"Taux d'erreur global : {total_errors}/{len(y_test)} = {total_errors/len(y_test)*100:.2f}%")

#%%
# -- 2.1 Chargement des donnees transformees --
# notebook.ipynb exporte : donnees_transformees.npz (StandardScaler + OHE)

data = np.load('donnees_transformees.npz', allow_pickle=True)
X_train_transformed = data['X_train']
X_test_transformed = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
feature_names_all = data['feature_names'].tolist()

rf_baseline = RandomForestClassifier(
    n_estimators=100,           # 100 arbres (défaut scikit-learn)
    class_weight='balanced',    # pondération automatique des classes
    random_state=2026,
    n_jobs=-1
)

rf_baseline.fit(X_train_transformed, y_train)
y_pred_baseline = rf_baseline.predict(X_test_transformed)

# Métriques baseline
acc_baseline  = accuracy_score(y_test, y_pred_baseline)
f1_baseline   = f1_score(y_test, y_pred_baseline, average='macro')
prec_baseline = precision_score(y_test, y_pred_baseline, average='macro')
rec_baseline  = recall_score(y_test, y_pred_baseline, average='macro')

# Importance des features
importances = rf_baseline.feature_importances_
indices = np.argsort(importances)[::-1]

# ── 5.1 RandomizedSearchCV — Optimisation des hyperparamètres ──
# Réf. : Bergstra & Bengio (2012). Random Search for Hyper-Parameter Optimization. JMLR.

param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample'],
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=2026, n_jobs=-1),
    param_distributions=param_distributions,
    n_iter=50,                # 50 combinaisons aléatoires
    scoring='f1_macro',
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=2026),
    n_jobs=-1,
    verbose=1,
    random_state=2026,
    return_train_score=True
)

total_fits = 50 * 3
print(f"Combinaisons testées : 50  |  Fits totaux : {total_fits}")
print(f"\nLancement du RandomizedSearchCV...")

random_search.fit(X_train_transformed, y_train)


# ── 6.1 Selection de features basee sur l'importance ──
# On garde les features qui captent X% de l'importance cumulee

# Recuperer le meilleur modele issu du RandomizedSearchCV
best_rf = random_search.best_estimator_

importances_best = best_rf.feature_importances_
indices_best = np.argsort(importances_best)[::-1]
cum_imp = np.cumsum(importances_best[indices_best])

# Seuils a tester : 80%, 85%, 90%, 95%
thresholds_feat = [0.80, 0.85, 0.90, 0.95]
cv_strat = StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)


results_feat_sel = []
for thresh in thresholds_feat:
    n_keep = np.argmax(cum_imp >= thresh) + 1
    selected_idx = indices_best[:n_keep]
    
    X_train_sel = X_train_transformed[:, selected_idx]
    X_test_sel  = X_test_transformed[:, selected_idx]
    
    # CV avec les memes params optimaux
    rf_sel = RandomForestClassifier(
        **random_search.best_params_,
        random_state=2026, n_jobs=-1
    )
    scores = cross_val_score(rf_sel, X_train_sel, y_train,
                             cv=cv_strat, scoring='f1_macro', n_jobs=-1)
    
    results_feat_sel.append({
        'seuil': thresh,
        'n_features': n_keep,
        'f1_mean': scores.mean(),
        'f1_std': scores.std(),
        'selected_idx': selected_idx
    })
    print(f"  {thresh*100:.0f}%{'':<6} {n_keep:<15} {scores.mean():.4f}{'':<8} {scores.std():.4f}")

# Meilleur seuil
best_feat_result = max(results_feat_sel, key=lambda x: x['f1_mean'])
sel_idx = best_feat_result['selected_idx']

print(f"\n  >> Meilleur : {best_feat_result['seuil']*100:.0f}% "
      f"({len(sel_idx)} features) "
      f"→ F1 = {best_feat_result['f1_mean']:.4f}")
print(f"  >> Ref. sans selection (185 features) → F1 = {random_search.best_score_:.4f}")

# ── 6.2 Threshold Tuning — Optimisation du seuil de decision ──
# Au lieu de predict() (seuil=0.5), on utilise predict_proba() pour trouver
# le seuil qui maximise le F1-Score macro sur un set de validation (CV)

from sklearn.model_selection import cross_val_predict

# Probabilites predites en cross-validation sur le train set
y_proba_cv = cross_val_predict(
    best_rf, X_train_transformed, y_train,
    cv=cv_strat, method='predict_proba', n_jobs=-1
)

# Tester differents seuils de 0.20 a 0.80
thresholds = np.arange(0.20, 0.81, 0.01)
f1_scores_thresh = []

for t in thresholds:
    y_pred_t = (y_proba_cv[:, 1] >= t).astype(int)
    f1_t = f1_score(y_train, y_pred_t, average='macro')
    f1_scores_thresh.append(f1_t)

f1_scores_thresh = np.array(f1_scores_thresh)
best_threshold = thresholds[np.argmax(f1_scores_thresh)]
best_f1_thresh = f1_scores_thresh.max()

print("=" * 65)
print("   THRESHOLD TUNING — Optimisation du seuil de decision")
print("=" * 65)
print(f"\n  Seuil par defaut : 0.50 → F1 macro (CV) = "
      f"{f1_score(y_train, (y_proba_cv[:, 1] >= 0.50).astype(int), average='macro'):.4f}")
print(f"  Seuil optimal    : {best_threshold:.2f} → F1 macro (CV) = {best_f1_thresh:.4f}")

# ── 7. MODELE OPTIMISE : Hyperparametres + Threshold Tuning ──

# Entrainer le modele optimise sur les features selectionnees uniquement
rf_optimized = RandomForestClassifier(
    **random_search.best_params_,
    random_state=2026, n_jobs=-1
)
rf_optimized.fit(X_train_transformed[:, sel_idx], y_train)

# Predire avec le seuil optimise sur les features selectionnees
y_proba_test = rf_optimized.predict_proba(X_test_transformed[:, sel_idx])
y_pred_optimized = (y_proba_test[:, 1] >= best_threshold).astype(int)

# Metriques
acc_optimized  = accuracy_score(y_test, y_pred_optimized)
f1_optimized   = f1_score(y_test, y_pred_optimized, average='macro')
prec_optimized = precision_score(y_test, y_pred_optimized, average='macro')
rec_optimized  = recall_score(y_test, y_pred_optimized, average='macro')

print("=" * 75)
print("   MODELE OPTIMISE — Random Forest")
print("=" * 75)
print(f"\n  Configuration :")
print(f"    - n_estimators    : {random_search.best_params_['n_estimators']}")
print(f"    - max_depth       : {random_search.best_params_['max_depth']}")
print(f"    - class_weight    : {random_search.best_params_['class_weight']}")
print(f"    - Seuil decision  : {best_threshold:.2f}")
print(f"\n  Performances sur le set de test :")
print(f"    - Accuracy        : {acc_optimized:.4f}")
print(f"    - Macro Precision : {prec_optimized:.4f}")
print(f"    - Macro Recall    : {rec_optimized:.4f}")
print(f"    - Macro F1-Score  : {f1_optimized:.4f}")
print(f"\n{'─' * 75}")
print("\nRapport de classification :")
print(classification_report(
    y_test, y_pred_optimized,
    target_names=TARGET_NAMES
))


# ── 8. COMPARAISON BASELINE vs OPTIMISE ──

# Déterminer le meilleur modèle
if f1_optimized > f1_baseline:
    best_model = "RF Optimise"
    best_acc = acc_optimized
    best_f1 = f1_optimized
    rf_best = rf_optimized
    y_pred_best = y_pred_optimized
else:
    best_model = "RF Baseline"
    best_acc = acc_baseline
    best_f1 = f1_baseline
    rf_best = rf_baseline
    y_pred_best = y_pred_baseline

# Tableau récapitulatif
print("\n" + "=" * 70)
print("   COMPARAISON BASELINE vs OPTIMISE")
print("=" * 70)
print(f"\n  {'Modele':<20} {'Accuracy':<15} {'Macro F1':<15}")
print("-" * 50)
print(f"  {'RF Baseline':<20} {acc_baseline:<15.4f} {f1_baseline:<15.4f}")
print(f"  {'RF Optimise':<20} {acc_optimized:<15.4f} {f1_optimized:<15.4f}")
print("-" * 50)
print(f"\n  >> MEILLEUR MODELE : {best_model}")
print(f"  >> Accuracy = {best_acc:.4f}, Macro F1 = {best_f1:.4f}")
print(f"  >> Gain F1 vs Baseline : +{(f1_optimized - f1_baseline)*100:.2f}%")

# ── 9. MODELE FINAL RETENU ──

print("=" * 70)
print(f"   MODELE RETENU : {best_model.upper()}")
print("=" * 70)


print(f"\nRapport de classification :")
print(classification_report(y_test, y_pred_best, target_names=TARGET_NAMES))

# %%
# ── EXPORT DES MODELES VERS models/ ──
os.makedirs('models', exist_ok=True)

# --- REGRESSION LOGISTIQUE : meilleur modele + metadata ---
joblib.dump(best_regression, 'models/best_regression.joblib')

reg_metadata = {
    "model_name": "Regression Logistique",
    "pca_components": int(pca.n_components_),
    "target_mapping": {k: int(v) for k, v in CIBLE_MAP.items()},
    "training_metrics": {
        "precision": float(reg_prec),
        "recall": float(reg_rec),
        "f1_macro": float(reg_f1)
    }
}
with open('models/reg_metadata.json', 'w') as f:
    json.dump(reg_metadata, f, indent=4)

# --- KNN : meilleur modele + PCA + preprocesseur ---
joblib.dump(best_knn, 'models/best_knn.joblib')

joblib.dump(pca, 'models/pca.joblib')

joblib.dump(joblib.load('preprocessor.joblib'), 'models/preprocessor.joblib')

knn_metadata = {
    "model_name": "KNN Optimise",
    "k_neighbors": int(best_knn.n_neighbors),
    "pca_components": int(pca.n_components_),
    "target_mapping": {k: int(v) for k, v in CIBLE_MAP.items()},
    "training_metrics": {
        "precision": float(prec_best),
        "recall": float(rec_best),
        "f1_macro": float(f1_best)
    }
}
with open('models/knn_metadata.json', 'w') as f:
    json.dump(knn_metadata, f, indent=4)

# --- Random Forest : meilleur modele + preprocesseur + metadata ---
joblib.dump(rf_best, 'models/rf_model.joblib')

joblib.dump(joblib.load('preprocessor.joblib'), 'models/rf_preprocessor.joblib')
sel_idx = best_feat_result['selected_idx']

# Métriques du meilleur modèle RF
if best_model == "RF Optimise":
    rf_acc, rf_prec, rf_rec, rf_f1 = acc_optimized, prec_optimized, rec_optimized, f1_optimized
else:
    rf_acc, rf_prec, rf_rec, rf_f1 = acc_baseline, prec_baseline, rec_baseline, f1_baseline

rf_metadata = {
    "model_name": best_model,
    "n_features_original": X_train_transformed.shape[1],
    "threshold": float(best_threshold),
    "target_mapping": CIBLE_MAP,
    "target_names": TARGET_NAMES,
    "training_metrics": {
        "precision": float(rf_prec),
        "recall": float(rf_rec),
        "f1_macro": float(rf_f1)
    }
}

# On n'ajoute les indices de selection que si le modele optimise a ete retenu
if best_model == "RF Optimise":
    rf_metadata["selected_features_indices"] = sel_idx.tolist()
    rf_metadata["n_features_selected"] = len(sel_idx)

with open('models/rf_metadata.json', 'w') as f:
    json.dump(rf_metadata, f, indent=4)
