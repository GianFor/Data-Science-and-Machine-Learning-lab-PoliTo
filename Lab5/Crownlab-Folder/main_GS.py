# write here your full pipeline code, no external files except `development.csv` and `evaluation.csv` should be used. # write here your full pipeline code, no external files except `development.csv` and `evaluation.csv` should be used. 
import os, random, numpy as np, torch, pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV





SEED = 42

# Python built - in random
random.seed(SEED)
# NumPy random
np.random.seed(SEED)

# Torch random (CPU and GPU)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Ensure deterministic behavior in CUDA operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

df = pd.read_csv("development.csv")

print("Dataset Description")
print("=======================================================================================")
print(df.describe(include="all"))

print("Dataset data types")
print("=======================================================================================")
print(df.dtypes)

print("Number of missing vals")
print("=======================================================================================")
print(df.dtypes)
print(df.isna().sum().sort_values(ascending = False))


print("Unique values for each col, excluding the missing ones")
print(df.nunique().sort_values())

print("\nValori delle colonne ordinali e categoriche")
for col in df.select_dtypes(include="object").columns:
    print(f"{col}: {sorted(df[col].dropna().unique())}")

X = df.drop(columns=["target"]).copy()
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

numeric_cols = X_train.select_dtypes(include="number").columns

median = X_train[numeric_cols].median()

X_train[numeric_cols] = X_train[numeric_cols].fillna(median)
X_test[numeric_cols] = X_test[numeric_cols].fillna(median)

ordinal_cols = X_train.filter(regex=r"^ord_").columns

for col in ordinal_cols:
    X_train[col] = X_train[col].str.extract(r"_(\d+)$", expand=False).astype(float)
    X_test[col] = X_test[col].str.extract(r"_(\d+)$", expand=False).astype(float)

ordinal_modes = X_train[ordinal_cols].mode().iloc[0]

X_train[ordinal_cols] = X_train[ordinal_cols].fillna(ordinal_modes)
X_test[ordinal_cols] = X_test[ordinal_cols].fillna(ordinal_modes)

cat_cols = X_train.filter(regex=r"^cat_").columns

train_cat = pd.get_dummies(X_train[cat_cols].fillna("missing"), dtype=int)
test_cat = pd.get_dummies(X_test[cat_cols].fillna("missing"), dtype=int)

test_cat = test_cat.reindex(columns=train_cat.columns, fill_value=0)

X_train = pd.concat(
    [X_train.drop(columns=cat_cols), train_cat], axis=1
)

X_test = pd.concat(
    [X_test.drop(columns=cat_cols), test_cat], axis=1
)

params = {
    "max_depth" : [10, 15, 20],
    "n_estimators" : [50, 100, 150],
}

rndf = RandomForestRegressor(criterion="squared_error", random_state=SEED)

search = GridSearchCV(
    estimator=rndf,
    param_grid=params,
    scoring="r2",
    cv=3,
    n_jobs=-1,
    refit=True,
    verbose=1
)

search.fit(X_train, y_train)

print("Migliori parametri:", search.best_params_)
print(f"R2 medio cross-validation: {search.best_score_:.4f}")

best_model = search.best_estimator_


y_pred = best_model.predict(X_test)

#best params are {'max_depth': 20, 'n_estimators': 150}
#R2 mean cross-validation: 0.7666
#Test R2: 0.6795369639473526

print(f"Test R2: {r2_score(y_pred, y_test)}")