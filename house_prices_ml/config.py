from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent

load_dotenv(Path(PROJECT_ROOT, ".env"))


# Paths

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"


# Target
TARGET = "SalePrice"


# Features

# Categorical features that are numerical
NUM_AS_CAT_COLS = [
    "MSSubClass",
    "OverallQual",
    "OverallCond",
]

# Features that defines NA as a value
NONE_FILL_COLS = [
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
]

# Ordinal features
ORDINAL_MAPPINGS = {
    "Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "OverallQual": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    "OverallCond": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "PoolQC": ["None", "Fa", "TA", "Gd", "Ex"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
    "PavedDrive": ["N", "P", "Y"],
    "LotShape": ["IR3", "IR2", "IR1", "Reg"],
}


# Threshold for categorical features with high cardinality
HIGH_CARDINALITY_THRESHOLD = 10
