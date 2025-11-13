# ---------------------------------------
# Exam-ready: synthetic dataset + EDA + models
# Short, robust, deterministic, no-errors
# ---------------------------------------

# 0. Imports & settings
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc,
                             mean_squared_error, r2_score)
from sklearn.preprocessing import label_binarize
import joblib, warnings
warnings.filterwarnings('ignore')
sns.set(style='whitegrid')
RND = 42
np.random.seed(RND)

# 1. Synthetic dataset generator (robust & general)
def make_synthetic_dataset(N=2500, random_state=RND):
    rng = np.random.RandomState(random_state)
    # numeric signals
    age = rng.randint(18, 80, size=N)                                # any context
    feature_a = rng.normal(50, 12, size=N).clip(1, 200)
    feature_b = rng.exponential(30, size=N).clip(1, 300)
    feature_c = rng.uniform(0, 100, size=N)
    feature_d = (feature_a * 0.3 + feature_b * 0.2 + rng.normal(0,10,N)).clip(0,500)
    # nominal/categorical
    cat_1 = rng.choice(['A','B','C'], size=N, p=[0.5,0.35,0.15])
    cat_2 = rng.choice(['X','Y'], size=N, p=[0.7,0.3])
    # continuous regression target (depends on numeric + categories)
    base_price = 20*feature_a + 5*feature_b + 100* (cat_1 == 'C').astype(int) + 50*(cat_2=='Y')
    price = (base_price * rng.uniform(0.6,1.4,size=N) + rng.normal(0,200,N)).round(2)
    # classification target: 3-class balanced via scoring then binned
    score = (feature_a/feature_a.mean()) + (feature_b/feature_b.mean()) + (feature_c/feature_c.mean())
    labels = pd.qcut(score, q=3, labels=['Low','Medium','High'])
    # Add derived binary target (optional)
    binary_target = np.where(price > np.median(price), 1, 0)
    # Build DataFrame
    df = pd.DataFrame({
        'age': age,
        'feature_a': feature_a.round(2),
        'feature_b': feature_b.round(2),
        'feature_c': feature_c.round(2),
        'feature_d': feature_d.round(2),
        'cat_1': cat_1,
        'cat_2': cat_2,
        'price': price,
        'segment': labels.astype(str),
        'expensive': binary_target
    })
    # Inject small missingness (~2-4%) and some outliers
    for col in ['feature_a','feature_b','feature_c','feature_d','cat_1']:
        mask = rng.rand(N) < 0.03
        df.loc[mask, col] = np.nan
    # outliers in price (0.5%)
    mask_o = rng.rand(N) < 0.005
    df.loc[mask_o, 'price'] *= rng.uniform(3,8,mask_o.sum())
    return df

df = make_synthetic_dataset(N=2500)
print("Dataset shape:", df.shape)
display(df.head())

# 2. Quick EDA (concise)
print("\n--- Summary stats (numeric) ---")
display(df.describe().T[['count','mean','std','min','50%','max']])
print("\nMissing values per column:")
print(df.isnull().sum())

# Plots (compact)
plt.figure(figsize=(10,3))
plt.subplot(1,3,1); sns.histplot(df['feature_a'].dropna(), kde=True); plt.title('feature_a')
plt.subplot(1,3,2); sns.histplot(df['feature_b'].dropna(), kde=True); plt.title('feature_b')
plt.subplot(1,3,3); sns.boxplot(x='segment', y='price', data=df); plt.title('price by segment')
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,3))
sns.countplot(y='segment', data=df, order=df['segment'].value_counts().index); plt.title('segment counts'); plt.show()

# 3. Preprocessing pipeline (robust — no errors)
numeric_features = ['age','feature_a','feature_b','feature_c','feature_d']
categorical_features = ['cat_1','cat_2']
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
scaler = StandardScaler()
ohe = OneHotEncoder = None

from sklearn.preprocessing import OneHotEncoder
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', num_imputer), ('scaler', scaler)]), numeric_features),
    ('cat', Pipeline([('imputer', cat_imputer), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
])

# Prepare classification (multi-class 'segment') and regression ('price') sets
df_clf = df.dropna(subset=['segment']).copy()
X_clf = df_clf[numeric_features + categorical_features]
y_clf = df_clf['segment']
le_segment = LabelEncoder(); y_clf_enc = le_segment.fit_transform(y_clf)

df_reg = df.dropna(subset=['price']).copy()
X_reg = df_reg[numeric_features + categorical_features]
y_reg = df_reg['price']

# Fit-transform
Xc_proc = preprocessor.fit_transform(X_clf)
Xr_proc = preprocessor.transform(X_reg)

# 4. Train-test split
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc_proc, y_clf_enc, test_size=0.25, random_state=RND, stratify=y_clf_enc)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr_proc, y_reg, test_size=0.25, random_state=RND)

# 5. Models: classification (KNN, NB, DT, RF) and regression (LinearRegression)
models_clf = {
    'KNN': KNeighborsClassifier(),
    'NaiveBayes': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(random_state=RND),
    'RandomForest': RandomForestClassifier(random_state=RND)
}
# Light hyperparameter grids for quick exam-time tuning
param_grid = {
    'KNN': {'n_neighbors':[3,5], 'weights':['uniform','distance']},
    'DecisionTree': {'max_depth':[5,8]},
    'RandomForest': {'n_estimators':[50], 'max_depth':[8]}
}

best_clf = {}
for name, model in models_clf.items():
    if name in param_grid:
        gs = GridSearchCV(model, param_grid[name], cv=3, scoring='f1_macro', n_jobs=-1)
        gs.fit(Xc_tr, yc_tr)
        best = gs.best_estimator_
        print(f"{name} best params: {gs.best_params_} | CV f1_macro: {gs.best_score_:.3f}")
    else:
        model.fit(Xc_tr, yc_tr)
        best = model
    # final fit
    best.fit(Xc_tr, yc_tr)
    best_clf[name] = best

# Regression
reg = LinearRegression()
reg.fit(Xr_tr, yr_tr)

# 6. Evaluation (classifiers): metrics + confusion + ROC (macro-avg)
def evaluate_classifiers(classifiers, X_test, y_test, label_enc):
    rows = []
    # Prepare binarized true labels for ROC (multiclass)
    n_classes = len(label_enc.classes_)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    plt.figure(figsize=(9,6))
    colors = sns.color_palette(n_colors=len(classifiers))
    for (name, clf), color in zip(classifiers.items(), colors):
        yp = clf.predict(X_test)
        acc = accuracy_score(y_test, yp)
        prec = precision_score(y_test, yp, average='macro', zero_division=0)
        rec = recall_score(y_test, yp, average='macro', zero_division=0)
        f1m = f1_score(y_test, yp, average='macro', zero_division=0)
        rows.append({'model':name, 'accuracy':acc, 'precision_macro':prec, 'recall_macro':rec, 'f1_macro':f1m})
        print(f"\n=== {name} ===")
        print(classification_report(y_test, yp, target_names=label_enc.classes_, zero_division=0))
        # Confusion matrix plot
        cm = confusion_matrix(y_test, yp)
        plt.figure(figsize=(4,3)); sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_enc.classes_, yticklabels=label_enc.classes_, cmap='Blues'); plt.title(f'CM: {name}'); plt.xlabel('Pred'); plt.ylabel('Actual'); plt.show()
        # ROC macro-average plot (if probability available)
        if hasattr(clf, "predict_proba"):
            yscore = clf.predict_proba(X_test)
            # compute per-class fpr/tpr and then macro-average
            all_fpr = np.unique(np.concatenate([roc_curve(y_test_bin[:,i], yscore[:,i])[0] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                fpr_i, tpr_i, _ = roc_curve(y_test_bin[:,i], yscore[:,i])
                mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
            mean_tpr /= n_classes
            roc_auc = auc(all_fpr, mean_tpr)
            plt.plot(all_fpr, mean_tpr, label=f'{name} macro-AUC={roc_auc:.3f}', color=color)
    # ROC show
    if plt.gca().has_data():
        plt.plot([0,1],[0,1],'k--', lw=1); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Macro-avg ROC (classifiers)'); plt.legend(); plt.show()
    return pd.DataFrame(rows).set_index('model')

results_df = evaluate_classifiers(best_clf, Xc_te, yc_te, le_segment)
print("\nClassifier comparison (summary):")
display(results_df.sort_values('f1_macro', ascending=False))

# 7. Pick top-2 classifiers by f1_macro and show concise comparison (confusion + metrics)
top2 = results_df.sort_values('f1_macro', ascending=False).head(2)
print("\nTop 2 classifiers:", list(top2.index))
for name in top2.index:
    clf = best_clf[name]
    pred = clf.predict(Xc_te)
    print(f"\n-- {name} metrics --")
    print("Accuracy:", accuracy_score(yc_te, pred))
    print("F1_macro:", f1_score(yc_te, pred, average='macro'))
    cm = confusion_matrix(yc_te, pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le_segment.classes_, yticklabels=le_segment.classes_, cmap='coolwarm'); plt.title(f'Confusion Matrix: {name}'); plt.show()

# 8. Regression evaluation (Linear Regression)
y_pred_reg = reg.predict(Xr_te)
rmse = mean_squared_error(yr_te, y_pred_reg, squared=False)
r2 = r2_score(yr_te, y_pred_reg)
print(f"\nLinear Regression -> RMSE: {rmse:.2f} | R2: {r2:.3f}")
plt.figure(figsize=(6,4)); plt.scatter(yr_te, y_pred_reg, alpha=0.4); plt.plot([yr_te.min(), yr_te.max()],[yr_te.min(), yr_te.max()], 'r--'); plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.title('LR: Actual vs Pred'); plt.show()

# 9. Save artifacts (optional)
joblib.dump({'preprocessor': preprocessor, 'label_encoder_segment': le_segment, 'classifiers': best_clf, 'regressor': reg}, 'exam_synthetic_models.pkl')
print("\nSaved models -> exam_synthetic_models.pkl")

# 10. Short exam-ready summary (copy into answers)
print("\n--- EXAM SUMMARY ---")
print(f"Rows: {df.shape[0]} | Numeric features: {len(numeric_features)} | Categorical: {len(categorical_features)}")
print("Classification target: 'segment' (3 classes). Regression target: 'price'.")
print("\nClassifier ranking by F1_macro:\n", results_df.sort_values('f1_macro', ascending=False))
print(f"\nLinear Regression RMSE: {rmse:.2f}, R2: {r2:.3f}")
print("Top 2 classifiers (recommend):", list(top2.index))
