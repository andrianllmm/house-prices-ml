import joblib
from loguru import logger
import pandas as pd
import typer

from house_prices_ml.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

OUTPUT_FILENAME = "submission.csv"

app = typer.Typer()


@app.command()
def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading test data from " + str(RAW_DATA_DIR / "test.csv"))
    test = pd.read_csv(RAW_DATA_DIR / "test.csv", index_col='Id')
    model = joblib.load(MODELS_DIR / "model.joblib")

    logger.info("Predicting...")
    predictions = model.predict(test)

    logger.info("Generating submission...")
    submission = pd.DataFrame({"Id": test.index, "SalePrice": predictions})
    submission.to_csv(PROCESSED_DATA_DIR / OUTPUT_FILENAME, index=False)

    logger.success("Submission generated at " + str(PROCESSED_DATA_DIR / OUTPUT_FILENAME))


if __name__ == "__main__":
    main()
