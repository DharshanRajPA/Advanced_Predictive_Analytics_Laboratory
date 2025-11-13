# -------------------------
# Synthetic dataset + EDA + Models
# Short, robust, exam-ready
# -------------------------

# 0. Imports & settings
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, roc_curve, auc, mean_squared_error, r2_score
)
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')
sns.set(style='whitegrid')
RND = 42
np.random.seed(RND)

# 1. Create a robust, general synthetic dataset (N rows)
def make_synthetic_cars(N=3000, random_state=RND):
    rng = np.random.RandomState(random_state)
    # numeric features
    cylinders = rng.choice([3,4,6,8], size=N, p=[0.05,0.7,0.2,0.05])
    displacement = (cylinders * rng.normal(250, 40, N)).clip(600, 6000)  # cc
    horsepower = (displacement/15 + rng.normal(0,20,N)).clip(40,700)
    torque = (horsepower * rng.uniform(0.7,1.2,N)).clip(30,1200)
    fuel_eff = (rng.normal(18,4,N) - (displacement/2000)).clip(4,30)  # kmpl
    seating = rng.choice([2,4,5,7,8], size=N, p=[0.02,0.05,0.8,0.1,0.03])
    doors = rng.choice([2,3,4,5], size=N, p=[0.05,0.05,0.8,0.1])
    airbags = rng.choice([0,1,2,4,6,8], size=N, p=[0.02,0.05,0.2,0.6,0.08,0.05])
    usb_ports = rng.choice([0,1,2,3,4], size=N, p=[0.05,0.15,0.6,0.15,0.05])
    # categorical features
    drivetrain = rng.choice(['FWD','RWD','AWD','4WD'], size=N, p=[0.6,0.15,0.2,0.05])
    fuel_type = rng.choice(['Petrol','Diesel','Hybrid','Electric'], size=N, p=[0.6,0.25,0.08,0.07])
    transmission = rng.choice(['Manual','Automatic','CVT'], size=N, p=[0.4,0.55,0.05])
    # continuous price (base on displacement, hp, features)
    base_price = (displacement * 120) + (horsepower * 300) + (airbags * 2000) + (usb_ports*500)
    # add categorical surcharges
    fuel_surcharge = np.where(fuel_type=='Electric', 150000, np.where(fuel_type=='Hybrid', 60000, 0))
    drive_surcharge = np.where(drivetrain=='AWD', 30000, np.where(drivetrain=='4WD', 40000, 0))
    price = (base_price + fuel_surcharge + drive_surcharge) * rng.uniform(0.7,1.5,N)
    price = price/10  # scale down to reasonable numbers
    # Introduce noise, outliers and missingness
    # Outliers: 0.5% very high price
    out_mask = rng.rand(N) < 0.005
    price[out_mask] *= rng.uniform(3,8, out_mask.sum())
    # Build DataFrame
    df = pd.DataFrame({
        'cylinders': cylinders,
        'displacement_cc': displacement.round(1),
        'horsepower': horsepower.round(1),
        'torque': torque.round(1),
        'fuel_eff_kmpl': fuel_eff.round(2),
        'seating': seating,
        'doors': doors,
        'airbags': airbags,
        'usb_ports': usb_ports,
        'drivetrain': drivetrain,
        'fuel_type': fuel_type,
        'transmission': transmission,
        'price': price.round(2)
    })
    # Introduce random missing values ~3% per column
    for col in df.columns:
        miss_idx = rng.rand(N) < 0.03
        df.loc[miss_idx, col] = np.nan
    # Create classification target: vehicle_segment (multi-class)
    # strategy: cluster by price and fuel efficiency
    # create score and cut into 4 classes
    score = (df['price'].fillna(df['price'].median())/df['price'].median()) + (df['fuel_eff_kmpl'].fillna(15)/15)
    df['segment'] = pd.qcut(score, q=4, labels=['Economy','Compact','Midsize','Premium'])
    # Create binary emission_class (derived)
    df['emission_class'] = np.where(df['fuel_type']=='Electric','Zero', np.where(df['fuel_eff_kmpl']>=18,'Low','High'))
    # Shuffle
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df

df = make_synthetic_cars(N=3000)
print("Synthetic dataset created:", df.shape)
display(df.head())
# Save to CSV (optional)
df.to_csv('synthetic_cars_generalized.csv', index=False)
print("Saved -> synthetic_cars_generalized.csv")

# 2. EDA (short & essential)
print("\n--- EDA: summary ---")
display(df.describe(include='all').T[['count','mean','std','min','50%','max']].head(12))
print("\nMissing per column:")
print(df.isnull().sum())

# Quick plots (histograms for numeric, counts for categorical)
num_cols = ['cylinders','displacement_cc','horsepower','torque','fuel_eff_kmpl','price']
cat_cols = ['drivetrain','fuel_type','transmission','segment','emission_class']
plt.figure(figsize=(12,5))
for i,c in enumerate(num_cols[:3]):
    plt.subplot(1,3,i+1); sns.histplot(df[c].dropna(), kde=True); plt.title(c)
plt.tight_layout(); plt.show()

plt.figure(figsize=(10,4))
for i,c in enumerate(cat_cols[:3]):
    plt.subplot(1,3,i+1); sns.countplot(y=df[c], order=df[c].value_counts().index); plt.title(c)
plt.tight_layout(); plt.show()

# Correlation heatmap (numeric only)
plt.figure(figsize=(7,5))
sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Numeric correlation"); plt.show()

# 3. Preprocessing pipeline (robust)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
numeric_features = ['cylinders','displacement_cc','horsepower','torque','fuel_eff_kmpl','seating','doors','airbags','usb_ports']
categorical_features = ['drivetrain','fuel_type','transmission']

# Imputers and scaler
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
scaler = StandardScaler()

# Prepare X, y for classification (segment) and regression (price)
# Encode target labels
clf_target = 'segment'   # multiclass: Economy/Compact/Midsize/Premium
reg_target = 'price'

# Clean: drop rows missing both price and segment
df_clean = df.copy().reset_index(drop=True)
df_clean = df_clean.dropna(subset=[clf_target, reg_target], how='all')  # keep rows with at least one target

# For classification task ensure no missing in target
df_clf = df_clean.dropna(subset=[clf_target]).copy()
df_reg = df_clean.dropna(subset=[reg_target]).copy()

# Feature matrix (use same features for both tasks)
X_clf = df_clf[numeric_features + categorical_features].copy()
y_clf = df_clf[clf_target].astype(str)
X_reg = df_reg[numeric_features + categorical_features].copy()
y_reg = df_reg[reg_target].astype(float)

# Encoding label for classification
le_segment = LabelEncoder()
y_clf_enc = le_segment.fit_transform(y_clf)

# Column transformer
from sklearn.preprocessing import OneHotEncoder
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', num_imputer), ('scaler', scaler)]), numeric_features),
    ('cat', Pipeline([('imputer', cat_imputer), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
], remainder='drop')

# Fit-transforms
X_clf_proc = preprocessor.fit_transform(X_clf)
X_reg_proc = preprocessor.transform(X_reg)

# 4. Train-test split
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clf_proc, y_clf_enc, test_size=0.25, random_state=RND, stratify=y_clf_enc)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg_proc, y_reg, test_size=0.25, random_state=RND)

# 5. Models: define simple configs & light grid-search for KNN and tree/RF
models_clf = {
    'KNN': KNeighborsClassifier(),
    'NaiveBayes': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(random_state=RND),
    'RandomForest': RandomForestClassifier(random_state=RND)
}
param_grids = {
    'KNN': {'n_neighbors':[3,5,7], 'weights':['uniform','distance']},
    'NaiveBayes': {},
    'DecisionTree': {'max_depth':[4,8,12], 'criterion':['gini','entropy']},
    'RandomForest': {'n_estimators':[50,100], 'max_depth':[6,10]}
}

best_clf = {}
for name, model in models_clf.items():
    if param_grids.get(name):
        gs = GridSearchCV(model, param_grids[name], cv=3, scoring='f1_macro', n_jobs=-1)
        gs.fit(Xc_train, yc_train)
        best = gs.best_estimator_
        print(f"{name} best params: {gs.best_params_}, CV f1_macro: {gs.best_score_:.3f}")
    else:
        model.fit(Xc_train, yc_train)
        best = model
    best.fit(Xc_train, yc_train)
    best_clf[name] = best

# 6. Regression model: Linear Regression (simple)
lr = LinearRegression()
lr.fit(Xr_train, yr_train)

# 7. Evaluation (classification models)
def eval_classifiers(classifiers, X_test, y_true, label_encoder):
    summary = []
    # get probabilities for ROC if available
    y_true_bin = label_binarize(y_true, classes=np.arange(len(label_encoder.classes_)))
    plt.figure(figsize=(10,8))
    colors = sns.color_palette(n_colors=len(classifiers))
    for (name, clf), c in zip(classifiers.items(), colors):
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        summary.append({'model':name,'accuracy':acc,'precision_macro':prec,'recall_macro':rec,'f1_macro':f1})
        print(f"\n--- {name} ---")
        print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, zero_division=0))
        # confusion matrix plot
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4,3)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_); plt.title(f'CM: {name}'); plt.show()
        # ROC (one-vs-rest) if predict_proba exists
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)
            # compute micro and macro AUC
            fpr = dict(); tpr = dict()
            for i in range(y_true_bin.shape[1]):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:,i], y_score[:,i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(y_true_bin.shape[1])]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(y_true_bin.shape[1]):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= y_true_bin.shape[1]
            roc_auc = auc(all_fpr, mean_tpr)
            plt.plot(all_fpr, mean_tpr, label=f'{name} macro-avg AUC={roc_auc:.3f}')
    # ROC legend
    if plt.gca().has_data():
        plt.plot([0,1],[0,1],'k--',lw=1); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Macro-avg ROC (classifiers)'); plt.legend(); plt.show()
    return pd.DataFrame(summary).set_index('model')

results_clf = eval_classifiers(best_clf, Xc_test, yc_test, le_segment)
print("\nClassifier summary:\n", results_clf)

# 8. Pick top two classifiers by f1_macro
top2 = results_clf.sort_values('f1_macro', ascending=False).head(2)
print("\nTop 2 classifiers by F1_macro:\n", top2)

# 9. Regression evaluation (Linear Regression)
y_pred_lr = lr.predict(Xr_test)
rmse = mean_squared_error(yr_test, y_pred_lr, squared=False)
r2 = r2_score(yr_test, y_pred_lr)
print(f"\nLinear Regression -> RMSE: {rmse:.2f}, R2: {r2:.3f}")
plt.figure(figsize=(6,4)); plt.scatter(yr_test, y_pred_lr, alpha=0.4); plt.plot([yr_test.min(), yr_test.max()],[yr_test.min(), yr_test.max()], 'r--'); plt.xlabel('Actual price'); plt.ylabel('Predicted price'); plt.title('LR: Actual vs Predicted'); plt.show()

# 10. Compare top2 classifiers more closely (ROC, confusion)
print("\nDetailed comparison of top 2 classifiers:")
for name in top2.index:
    clf = best_clf[name]
    print(f"\n--- {name} ---")
    # Confusion matrix
    y_pred = clf.predict(Xc_test)
    print("Accuracy:", accuracy_score(yc_test, y_pred), "F1_macro:", f1_score(yc_test, y_pred, average='macro'))
    cm = confusion_matrix(yc_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le_segment.classes_, yticklabels=le_segment.classes_); plt.title(f'Confusion Matrix: {name}'); plt.show()

# 11. Save models & preprocessor
import joblib
joblib.dump({'preprocessor':preprocessor, 'label_encoder_segment': le_segment, 'clf_models': best_clf, 'reg_model': lr}, 'models_synthetic_cars_joblib.pkl')
print("Saved models -> models_synthetic_cars_joblib.pkl")

# 12. Short summary output for exam report (printable)
print("\n--- SUMMARY (copy into report) ---")
print("Dataset rows:", df.shape[0], "| Features:", len(numeric_features)+len(categorical_features))
print("Classification target: 'segment' (4 classes). Regression target: 'price'.")
print("\nClassifier ranking by F1_macro:\n", results_clf.sort_values('f1_macro', ascending=False))
print(f"\nLinear Regression RMSE: {rmse:.2f}, R2: {r2:.3f}")
print("\nTop 2 classifiers (recommendation):", list(top2.index))
