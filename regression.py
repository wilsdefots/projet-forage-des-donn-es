# %%
# -- Bibliotheques standard --
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# -- Scikit-learn : modelisation --
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    GridSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV 

# -- Scikit-learn : metriques --
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score, f1_score,
    precision_score, recall_score
)

# -- Constantes --
CIBLE_MAP = {'BenignPositive': 0, 'FalsePositive': 1, 'TruePositive': 2}
CIBLE_MAP_INV = {v: k for k, v in CIBLE_MAP.items()}

print("Bibliotheques importees avec succes.")

#%%

# -- 2.1 Chargement du dataset pretraite --
df = pd.read_csv("donnees_preprocessees.csv", low_memory=False)
print(f"Dimensions : {df.shape[0]:,} lignes x {df.shape[1]} colonnes")

# Separation X / y
X = df.drop(columns=['cible'])
y = df['cible']

# Identifier les colonnes numeriques et categorielles
num_cols = X.select_dtypes(exclude='object').columns.tolist()
cat_cols = X.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype(str)

print(f"\n  Numeriques    ({len(num_cols)}) : {num_cols}")
print(f"  Categorielles ({len(cat_cols)}) : {cat_cols}")
print(f"\n  Distribution cible :")
print(y.map(CIBLE_MAP_INV).value_counts().to_string())
"""
"""
# -- 2.2 Split stratifie (80/20) --
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)
print(f"\n  Train : {X_train.shape[0]:,}  |  Test : {X_test.shape[0]:,}")

# -- 2.3 Transformation (StandardScaler + OneHotEncoder) --
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore',
                              sparse_output=False), cat_cols),
    ]
)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed  = preprocessor.transform(X_test)

print(f"\n  Dimensionnalite apres OneHotEncoding : {X_train_transformed.shape[1]} features")
print(f"  Train : {X_train_transformed.shape}  |  Test : {X_test_transformed.shape}")



#%%

# -- 3.1 Analyse de la variance expliquee --

# Fit PCA complet pour analyser la distribution de la variance
pca_full = PCA(random_state=2026)
pca_full.fit(X_train_transformed)

variance_cumulee = np.cumsum(pca_full.explained_variance_ratio_)

# Nombre de composantes pour 95% de variance
n_composantes_95 = np.argmax(variance_cumulee >= 0.95) + 1

print(f"Dimensions originales : {X_train_transformed.shape[1]} features")
print(f"Composantes pour 95% de variance : {n_composantes_95}")
print(f"Taux de compression : {(1 - n_composantes_95/X_train_transformed.shape[1])*100:.1f}%")
print(f"\nVariance expliquee par les 5 premieres composantes :")
for i in range(5):
    print(f"  PC{i+1} : {pca_full.explained_variance_ratio_[i]*100:.2f}%  (cumule : {variance_cumulee[i]*100:.2f}%)")


# -- 3.2 Visualisation de la variance cumulee --

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Graphique 1 : Variance cumulee
axes[0].plot(range(1, len(variance_cumulee) + 1), variance_cumulee * 100, 'b-', linewidth=2)
axes[0].axhline(y=95, color='red', linestyle='--', linewidth=1.5, label='Seuil 95%')
axes[0].axvline(x=n_composantes_95, color='green', linestyle='--', linewidth=1.5,
                label=f'{n_composantes_95} composantes')
axes[0].scatter([n_composantes_95], [95], color='red', s=100, zorder=5, edgecolors='black')
axes[0].set_xlabel('Nombre de composantes', fontsize=11)
axes[0].set_ylabel('Variance cumulee expliquee (%)', fontsize=11)
axes[0].set_title('Variance cumulee -- Selection du nombre de composantes',
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 105)

# Graphique 2 : Variance individuelle (top 30 composantes)
n_show = min(30, len(pca_full.explained_variance_ratio_))
axes[1].bar(range(1, n_show + 1), pca_full.explained_variance_ratio_[:n_show] * 100,
            color='steelblue', alpha=0.7)
axes[1].set_xlabel('Composante principale', fontsize=11)
axes[1].set_ylabel('Variance expliquee (%)', fontsize=11)
axes[1].set_title(f'Variance individuelle (top {n_show} composantes)',
                  fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('Analyse en Composantes Principales (ACP / PCA)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print(f"\nOn retient {n_composantes_95} composantes (95% de variance, "
      f"compression de {(1 - n_composantes_95/X_train_transformed.shape[1])*100:.0f}%)")


# -- 3.3 Application de la PCA (95% de variance) --

pca = PCA(n_components=0.95, random_state=2026)
X_train_pca = pca.fit_transform(X_train_transformed)
X_test_pca  = pca.transform(X_test_transformed)

print(f"Transformation PCA appliquee :")
print(f"  Train : {X_train_transformed.shape} -> {X_train_pca.shape}")
print(f"  Test  : {X_test_transformed.shape} -> {X_test_pca.shape}")
print(f"  Composantes retenues : {pca.n_components_}")
print(f"  Variance expliquee   : {pca.explained_variance_ratio_.sum()*100:.2f}%")
print(f"\nDonnees pretes pour la régression linéaire avec PCA.")

#%%


# -- 4.1 Régression logistique -- Sans PCA vs Avec PCA --

# --- Sans PCA (toutes les features) ---
logistic_baseline_nopca = LogisticRegression(
    max_iter=1000,
    random_state=42,
    multi_class='ovr'
)
logistic_baseline_nopca.fit(X_train_transformed, y_train)
y_pred_bl_nopca = logistic_baseline_nopca.predict(X_test_transformed)

acc_bl_nopca = accuracy_score(y_test, y_pred_bl_nopca)
f1_bl_nopca  = f1_score(y_test, y_pred_bl_nopca, average='macro')

# --- Avec PCA ---
logistic_baseline = LogisticRegression(
    max_iter=1000,
    random_state=42,
    multi_class='ovr'
)
logistic_baseline.fit(X_train_pca, y_train)
y_pred_baseline = logistic_baseline.predict(X_test_pca)

acc_baseline = accuracy_score(y_test, y_pred_baseline)
f1_baseline  = f1_score(y_test, y_pred_baseline, average='macro')

print("=" * 65)
print("        LOGISTIC REGRESSION BASELINE -- IMPACT DE LA PCA")
print("=" * 65)
print(f"\n  {'Configuration':<25} {'Accuracy':>10} {'Macro F1':>10} {'Features':>10}")
print(f"  {'-' * 55}")
print(f"  {'Sans PCA':<25} {acc_bl_nopca:>10.4f} {f1_bl_nopca:>10.4f} {X_train_transformed.shape[1]:>10}")
print(f"  {'Avec PCA (95% var.)':<25} {acc_baseline:>10.4f} {f1_baseline:>10.4f} {X_train_pca.shape[1]:>10}")
print(f"\n  Delta F1 (PCA - Sans PCA) : {f1_baseline - f1_bl_nopca:+.4f}")
print(f"  Compression : {X_train_transformed.shape[1]} -> {X_train_pca.shape[1]} features "
      f"(-{(1 - X_train_pca.shape[1]/X_train_transformed.shape[1])*100:.0f}%)")
print(f"\n{'-' * 65}")
print("\nRapport de classification (Avec PCA) :")
print(classification_report(
    y_test, y_pred_baseline,
    target_names=['BenignPositive', 'FalsePositive', 'TruePositive']
))


# -- 4.2 Matrice de confusion -- Baseline avec PCA --
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_baseline,
    display_labels=['BenignPositive', 'FalsePositive', 'TruePositive'],
    cmap='Blues', ax=ax
)
ax.set_title(f'Matrice de confusion -- Régression Logistique (PCA {X_train_pca.shape[1]} comp.)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
# GridSearch prenait trop de temps à run, donc RandomizedSearch est utilisé à la place.
param_distributions = {
    'C': np.logspace(-3, 3, 20),
    'penalty': ['l1', 'l2'],
    'solver': ['saga'],
    'class_weight': [None, 'balanced']
}

log_reg = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')


grid_search = RandomizedSearchCV (
    estimator=log_reg,
    param_distributions=param_distributions,
    scoring='f1_macro',
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=2026),
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)


n_iter = 20
n_splits = 3
n_fits = n_iter * n_splits

print(f"Logistic regression hyperparameter tuning ({X_train_pca.shape[1]} composantes PCA, 3‑fold CV)")
print(f"Combinaisons testeés (n_iter) : {n_iter}  |  Fits totaux : {n_fits}")
print("\nLancement du RandomizedSearch...\n")

grid_search.fit(X_train_pca, y_train)

print("RandomizedSearchCV termine !")
print("=" * 60)
print("  Meilleurs hyperparametres :")
for param, val in grid_search.best_params_.items():
    print(f"    {param} = {val}")
print(f"\n  Meilleur Macro F1 (CV) : {grid_search.best_score_:.4f}")
print("=" * 60)

#%%
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

print("Top 15 des combinaisons d'hyperparametres :\n")
print(results_display.to_string(index=False))

# Verification du surapprentissage
best_train = results_df.loc[results_df['rank_test_score'] == 1, 'mean_train_score'].values[0]
best_test  = grid_search.best_score_
gap = best_train - best_test
print(f"\nEcart Train-Test (meilleur modele) : {gap:.4f}")
if gap > 0.05:
    print("   Ecart significatif -> risque de surapprentissage")
else:
    print("   Ecart acceptable -> pas de surapprentissage majeur")

#%%
# -- 7.1 Predictions avec le meilleur modele --

best_regression = grid_search.best_estimator_
y_pred_best = best_regression.predict(X_test_pca)

# Metriques detaillees
acc_best  = accuracy_score(y_test, y_pred_best)
f1_best   = f1_score(y_test, y_pred_best, average='macro')
prec_best = precision_score(y_test, y_pred_best, average='macro')
rec_best  = recall_score(y_test, y_pred_best, average='macro')

print("=" * 65)
print("     EVALUATION FINALE -- RÉGRESSION OPTIMISE + PCA (sur jeu de test)")
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
    target_names=['BenignPositive', 'FalsePositive', 'TruePositive']
))


# -- 7.2 Matrice de confusion -- Modele optimise --

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_best,
    display_labels=['BenignPositive', 'FalsePositive', 'TruePositive'],
    cmap='Blues', ax=axes[0]
)
axes[0].set_title('Matrice de confusion (valeurs absolues)', fontsize=12, fontweight='bold')

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_best,
    display_labels=['BenignPositive', 'FalsePositive', 'TruePositive'],
    normalize='true', cmap='Greens', values_format='.2%', ax=axes[1]
)
axes[1].set_title('Matrice de confusion (normalisee par classe)', fontsize=12, fontweight='bold')

plt.suptitle('Régression Optimisée -- Matrices de confusion sur le jeu de test',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

#%%

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
    'PCA + Optimise': [acc_best, prec_best, rec_best, f1_best]
})
comparison['Gain total'] = comparison['PCA + Optimise'] - comparison['Sans PCA']
comparison['Gain (%)'] = (comparison['Gain total'] / comparison['Sans PCA'] * 100).round(2)

print("=" * 90)
print("     COMPARAISON : SANS PCA  vs  PCA + BASELINE  vs  PCA + OPTIMISE")
print("=" * 90)
print(comparison.to_string(index=False))

# Visualisation
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(comparison))
width = 0.25

bars1 = ax.bar(x - width, comparison['Sans PCA'], width,
               label=f'Sans PCA', color='lightcoral')
bars2 = ax.bar(x, comparison['PCA + Baseline'], width,
               label=f'PCA + Baseline', color='lightskyblue')
bars3 = ax.bar(x + width, comparison['PCA + Optimise'], width,
               label='PCA + Optimise', color='steelblue')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Comparaison des performances : Sans PCA -> PCA + Baseline -> PCA + Optimise',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison['Metrique'], fontsize=10)
ax.legend(fontsize=10, loc='lower right')
ax.set_ylim(0, 1.05)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

#%%
# -- 9.1 Analyse des erreurs par classe --

cm = confusion_matrix(y_test, y_pred_best)
classes = ['BenignPositive', 'FalsePositive', 'TruePositive']

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

# -- 9.2 Distribution des predictions vs realite --

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution reelle
y_test.map(CIBLE_MAP_INV).value_counts().plot(kind='bar', ax=axes[0], color='steelblue', alpha=0.7)
axes[0].set_title('Distribution reelle (y_test)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Nombre')
axes[0].tick_params(axis='x', rotation=0)

# Distribution predite
pd.Series(y_pred_best).map(CIBLE_MAP_INV).value_counts().plot(kind='bar', ax=axes[1], color='coral', alpha=0.7)
axes[1].set_title('Distribution predite (y_pred)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Nombre')
axes[1].tick_params(axis='x', rotation=0)

plt.suptitle('Comparaison des distributions : reelle vs predite', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
