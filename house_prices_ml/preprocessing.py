import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder  # , StandardScaler

from house_prices_ml.config import (
    HIGH_CARDINALITY_THRESHOLD,
    NONE_FILL_COLS,
    NUM_AS_CAT_COLS,
    ORDINAL_MAPPINGS,
)


class TypeCaster(BaseEstimator, TransformerMixin):
    def __init__(self, num_as_cat_cols=NUM_AS_CAT_COLS):
        self.num_as_cat_cols = num_as_cat_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for col in self.num_as_cat_cols:
            if col in X.columns:
                X[col] = X[col].astype(str)

        self._feature_names_out_ = np.array(X.columns)
        return X

    def get_feature_names_out(self, input_features=None):
        return self._feature_names_out_


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.drop_cols_ = [
            "TotalBsmtSF",
            "1stFlrSF",
            "2ndFlrSF",
            "FullBath",
            "HalfBath",
            "BsmtFullBath",
            "BsmtHalfBath",
            "YrSold",
            "YearBuilt",
            "YearRemodAdd",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["TotalSF"] = X.get("TotalBsmtSF", 0) + X.get("1stFlrSF", 0) + X.get("2ndFlrSF", 0)

        X["TotalBath"] = (
            X.get("FullBath", 0)
            + 0.5 * X.get("HalfBath", 0)
            + X.get("BsmtFullBath", 0)
            + 0.5 * X.get("BsmtHalfBath", 0)
        )
        X["Age"] = X.get("YrSold", 0) - X.get("YearBuilt", 0)

        X["RemodAge"] = X.get("YrSold", 0) - X.get("YearRemodAdd", 0)

        X["IsRemodeled"] = (X["YearBuilt"] != X["YearRemodAdd"]).astype(int)

        X = X.drop(columns=[c for c in self.drop_cols_ if c in X.columns], errors="ignore")

        self._feature_names_out_ = np.array(X.columns)
        return X

    def get_feature_names_out(self, input_features=None):
        return self._feature_names_out_


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        if self.cols is None:
            self.cols = X.columns.tolist()

        self.freq_maps_ = {col: X[col].value_counts(normalize=True) for col in self.cols}

        self._feature_names_out_ = np.array(self.cols)
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            freq_map = self.freq_maps_.get(col, {})
            X[col] = X[col].map(freq_map).fillna(0)
        return X

    def get_feature_names_out(self, input_features=None):
        return self._feature_names_out_


def get_imputer():
    num_imputer = SimpleImputer(strategy="median").set_output(transform="pandas")
    cat_imputer = SimpleImputer(strategy="most_frequent").set_output(transform="pandas")
    none_imputer = SimpleImputer(strategy="constant", fill_value="None").set_output(
        transform="pandas"
    )

    imputer = ColumnTransformer(
        transformers=[
            ("num", num_imputer, select_num_cols),
            ("cat", cat_imputer, select_cat_cols_impute),
            ("none", none_imputer, select_none_cols),
        ],
        verbose_feature_names_out=False,
        remainder="passthrough",
    ).set_output(transform="pandas")

    return imputer


def get_encoder_scaler():
    onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(
        transform="pandas"
    )
    ordinal_encoder = OrdinalEncoder(
        categories=list(ORDINAL_MAPPINGS.values()),
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    ).set_output(transform="pandas")
    freq_encoder = FrequencyEncoder()

    # scaler = StandardScaler().set_output(transform="pandas")

    encoder_scaler = ColumnTransformer(
        transformers=[
            ("onehot", onehot_encoder, select_cat_cols_encode),
            ("ordinal", ordinal_encoder, select_ordinal_cols),
            ("freq", freq_encoder, select_freq_cols),
            # ("scale", scaler, select_num_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    return encoder_scaler


def get_preprocessor():
    preprocessor = Pipeline(
        [
            ("type_cast", TypeCaster()),
            ("feat_eng", FeatureEngineer()),
            ("impute", get_imputer()),
            ("encode_scale", get_encoder_scaler()),
        ]
    )

    return preprocessor


# Column selectors


def select_num_cols(df):
    return df.select_dtypes(exclude="object").columns.tolist()


def select_cat_cols_impute(df):
    return [c for c in df.select_dtypes(include="object").columns if c not in NONE_FILL_COLS]


def select_none_cols(df):
    return [c for c in NONE_FILL_COLS if c in df.columns and df[c].dtype == "object"]


def select_cat_cols_encode(df):
    return [
        c
        for c in df.select_dtypes(include="object").columns
        if c not in ORDINAL_MAPPINGS.keys() and df[c].nunique() <= HIGH_CARDINALITY_THRESHOLD
    ]


def select_ordinal_cols(df):
    return [
        c
        for c in ORDINAL_MAPPINGS.keys()
        if c in df.columns and df[c].nunique() <= HIGH_CARDINALITY_THRESHOLD
    ]


def select_freq_cols(df):
    return [
        c
        for c in df.select_dtypes(include="object")
        if df[c].nunique() > HIGH_CARDINALITY_THRESHOLD and c not in ORDINAL_MAPPINGS.keys()
    ]
