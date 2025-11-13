# Executing the exam-ready synthetic dataset notebook from earlier.
# This will generate the dataset, run EDA, train models, evaluate, and save artifacts.
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, joblib
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
import warnings, os
warnings.filterwarnings('ignore')
sns.set(style='whitegrid')
RND = 42
np.random.seed(RND)

def make_synthetic_dataset(N=2500, random_state=RND):
    rng = np.random.RandomState(random_state)
    age = rng.randint(18, 80, size=N)
    feature_a = rng.normal(50, 12, size=N).clip(1, 200)
    feature_b = rng.exponential(30, size=N).clip(1, 300)
    feature_c = rng.uniform(0, 100, size=N)
    feature_d = (feature_a * 0.3 + feature_b * 0.2 + rng.normal(0,10,N)).clip(0,500)
    cat_1 = rng.choice(['A','B','C'], size=N, p=[0.5,0.35,0.15])
    cat_2 = rng.choice(['X','Y'], size=N, p=[0.7,0.3])
    base_price = 20*feature_a + 5*feature_b + 100* (cat_1 == 'C').astype(int) + 50*(cat_2=='Y')
    price = (base_price * rng.uniform(0.6,1.4,size=N) + rng.normal(0,200,N)).round(2)
    score = (feature_a/feature_a.mean()) + (feature_b/feature_b.mean()) + (feature_c/feature_c.mean())
    labels = pd.qcut(score, q=3, labels=['Low','Medium','High'])
    binary_target = np.where(price > np.median(price), 1, 0)
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
    for col in ['feature_a','feature_b','feature_c','feature_d','cat_1']:
        mask = rng.rand(N) < 0.03
        df.loc[mask, col] = np.nan
    mask_o = rng.rand(N) < 0.005
    df.loc[mask_o, 'price'] *= rng.uniform(3,8,mask_o.sum())
    return df

# Generate dataset
df = make_synthetic_dataset(N=2500)
df.to_csv('/mnt/data/synthetic_cars_generalized.csv', index=False)

# EDA summary (store results)
eda_summary = df.describe(include='all').T
missing_counts = df.isnull().sum()

# Preprocessing
numeric_features = ['age','feature_a','feature_b','feature_c','feature_d']
categorical_features = ['cat_1','cat_2']
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
scaler = StandardScaler()
from sklearn.preprocessing import OneHotEncoder
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', num_imputer), ('scaler', scaler)]), numeric_features),
    ('cat', Pipeline([('imputer', cat_imputer), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
])

# Prepare datasets
df_clf = df.dropna(subset=['segment']).copy()
X_clf = df_clf[numeric_features + categorical_features]
y_clf = df_clf['segment']
le_segment = LabelEncoder(); y_clf_enc = le_segment.fit_transform(y_clf)

df_reg = df.dropna(subset=['price']).copy()
X_reg = df_reg[numeric_features + categorical_features]
y_reg = df_reg['price']

Xc_proc = preprocessor.fit_transform(X_clf)
Xr_proc = preprocessor.transform(X_reg)

Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc_proc, y_clf_enc, test_size=0.25, random_state=RND, stratify=y_clf_enc)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr_proc, y_reg, test_size=0.25, random_state=RND)

# Models and grid search
models_clf = {
    'KNN': KNeighborsClassifier(),
    'NaiveBayes': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(random_state=RND),
    'RandomForest': RandomForestClassifier(random_state=RND)
}
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
    else:
        model.fit(Xc_tr, yc_tr)
        best = model
    best.fit(Xc_tr, yc_tr)
    best_clf[name] = best

# Regression
reg = LinearRegression()
reg.fit(Xr_tr, yr_tr)

# Evaluate classifiers
results = []
n_classes = len(le_segment.classes_)
y_test_bin = label_binarize(yc_te, classes=range(n_classes))
roc_curves = {}
for name, clf in best_clf.items():
    yp = clf.predict(Xc_te)
    acc = accuracy_score(yc_te, yp)
    prec = precision_score(yc_te, yp, average='macro', zero_division=0)
    rec = recall_score(yc_te, yp, average='macro', zero_division=0)
    f1m = f1_score(yc_te, yp, average='macro', zero_division=0)
    cm = confusion_matrix(yc_te, yp)
    report = classification_report(yc_te, yp, target_names=le_segment.classes_, zero_division=0)
    results.append({'model':name,'accuracy':acc,'precision_macro':prec,'recall_macro':rec,'f1_macro':f1m,'confusion_matrix':cm,'report':report})
    if hasattr(clf, "predict_proba"):
        yscore = clf.predict_proba(Xc_te)
        all_fpr = np.unique(np.concatenate([roc_curve(y_test_bin[:,i], yscore[:,i])[0] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            fpr_i, tpr_i, _ = roc_curve(y_test_bin[:,i], yscore[:,i])
            mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
        mean_tpr /= n_classes
        roc_auc = auc(all_fpr, mean_tpr)
        roc_curves[name] = {'fpr': all_fpr, 'tpr': mean_tpr, 'auc': roc_auc}

results_df = pd.DataFrame(results).set_index('model')

# Regression eval
y_pred_reg = reg.predict(Xr_te)
rmse = mean_squared_error(yr_te, y_pred_reg, squared=False)
r2 = r2_score(yr_te, y_pred_reg)

# Save artifacts
joblib.dump({'preprocessor': preprocessor, 'label_encoder_segment': le_segment, 'classifiers': best_clf, 'regressor': reg}, '/mnt/data/exam_synthetic_models.pkl')
df.to_csv('/mnt/data/synthetic_cars_generalized.csv', index=False)

# Prepare outputs for display to user
summary = {
    'dataset_shape': df.shape,
    'head': df.head().to_dict(),
    'eda_summary_head': eda_summary.head().to_dict(),
    'missing_counts': missing_counts.to_dict(),
    'results_table': results_df.reset_index().to_dict(orient='records'),
    'roc_curves': {k: {'auc': v['auc']} for k,v in roc_curves.items()},
    'regression': {'rmse': rmse, 'r2': r2},
    'saved_files': ['/mnt/data/synthetic_cars_generalized.csv', '/mnt/data/exam_synthetic_models.pkl']
}

summary

