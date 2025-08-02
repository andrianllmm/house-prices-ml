import joblib
from loguru import logger
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import typer
from xgboost import XGBRegressor

from house_prices_ml.config import MODELS_DIR, RAW_DATA_DIR, TARGET
from house_prices_ml.modelling import build_pipeline, tune_hyperparameters

app = typer.Typer()


@app.command()
def train_model():
    logger.info("Loading training data...")
    train = pd.read_csv(RAW_DATA_DIR / "train.csv", index_col='Id')

    X = train.drop(TARGET, axis=1)
    y = train[TARGET]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info("Building pipeline...")
    model = XGBRegressor(random_state=42)
    pipeline = build_pipeline(model)

    logger.info("Tuning hyperparameters...")
    search = tune_hyperparameters(pipeline, X_train, y_train, verbose=2)

    logger.info("Best parameters found.")
    best_model = search.best_estimator_

    joblib.dump(best_model, MODELS_DIR / "model.joblib")
    logger.success(f"Model saved at {MODELS_DIR / 'model.joblib'}")

    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)

    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)

    logger.info(f"Train MSE: {train_mse:,.2f}")
    logger.info(f"Validation MSE: {val_mse:,.2f}")
    logger.info(f"Search best score: {search.best_score_:.4f}")


if __name__ == "__main__":
    app()
