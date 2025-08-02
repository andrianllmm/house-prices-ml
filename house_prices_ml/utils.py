from house_prices_ml.config import NUM_AS_CAT_COLS, TARGET


def fix_column_dtypes(X):
    X = X.copy()
    for col in NUM_AS_CAT_COLS:
        X[col] = X[col].astype(str)
    return X


def infer_cols_dtypes(X, exclude=[TARGET]):
    X = fix_column_dtypes(X)

    if exclude:
        X = X.drop(columns=exclude, errors="ignore")

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    return num_cols, cat_cols
