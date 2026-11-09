# Full pipeline: only development.csv and evaluation.csv are required.
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


SEED = 42


def preprocess(X_train, X_test):
    """Prepare both datasets using statistics from the development set only."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Continuous features: fill missing values with development medians.
    numeric_cols = X_train.select_dtypes(include="number").columns
    medians = X_train[numeric_cols].median()
    X_train[numeric_cols] = X_train[numeric_cols].fillna(medians)
    X_test[numeric_cols] = X_test[numeric_cols].fillna(medians)

    # Ordinal features: extract levels, then fill with development modes.
    ordinal_cols = X_train.filter(regex=r"^ord_").columns
    for col in ordinal_cols:
        X_train[col] = X_train[col].str.extract(r"_(\d+)$", expand=False).astype(float)
        X_test[col] = X_test[col].str.extract(r"_(\d+)$", expand=False).astype(float)

    ordinal_modes = X_train[ordinal_cols].mode().iloc[0]
    X_train[ordinal_cols] = X_train[ordinal_cols].fillna(ordinal_modes)
    X_test[ordinal_cols] = X_test[ordinal_cols].fillna(ordinal_modes)

    # Categorical features: one-hot encode and align evaluation columns.
    cat_cols = X_train.filter(regex=r"^cat_").columns
    train_cat = pd.get_dummies(X_train[cat_cols].fillna("missing"), dtype=int)
    test_cat = pd.get_dummies(X_test[cat_cols].fillna("missing"), dtype=int)
    test_cat = test_cat.reindex(columns=train_cat.columns, fill_value=0)

    X_train = pd.concat([X_train.drop(columns=cat_cols), train_cat], axis=1)
    X_test = pd.concat([X_test.drop(columns=cat_cols), test_cat], axis=1)
    return X_train, X_test


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    development = pd.read_csv("development.csv")
    evaluation = pd.read_csv("evaluation.csv")
    y_train = development["target"]
    X_train, X_test = preprocess(development.drop(columns="target"), evaluation)

    model = RandomForestRegressor(
        max_depth=20,
        n_estimators=150,
        random_state=SEED,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    submission = pd.DataFrame({"value": predictions})
    submission.to_csv("submission.csv", index_label="index")


if __name__ == "__main__":
    main()
