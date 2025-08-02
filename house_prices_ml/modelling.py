import numpy as np
from scipy.stats import randint, uniform
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from house_prices_ml.preprocessing import get_preprocessor


def build_pipeline(model):
    transformed_model = TransformedTargetRegressor(
        regressor=model, func=np.log1p, inverse_func=np.expm1
    )

    return Pipeline(
        [
            ("preprocessor", get_preprocessor()),
            ("model", transformed_model),
        ]
    )


def tune_hyperparameters(pipeline, X, y, verbose=1):
    param_dist = {
        "model__regressor__n_estimators": randint(300, 1200),
        "model__regressor__learning_rate": uniform(0.005, 0.05),
        "model__regressor__max_depth": randint(2, 8),
        "model__regressor__min_child_weight": randint(1, 10),
        "model__regressor__subsample": uniform(0.6, 0.4),
        "model__regressor__colsample_bytree": uniform(0.6, 0.4),
        "model__regressor__gamma": uniform(0, 5),
        "model__regressor__reg_alpha": uniform(0, 1),
        "model__regressor__reg_lambda": uniform(0.5, 1.5),
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        verbose=verbose,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X, y)

    return search
