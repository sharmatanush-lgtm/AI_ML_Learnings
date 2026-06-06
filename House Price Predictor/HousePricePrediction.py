import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("HousePricePrediction.csv")
df = df.drop_duplicates()
df["SalePrice"] = pd.to_numeric(df["SalePrice"], errors="coerce")
df["SalePrice"] = df["SalePrice"].fillna(df["SalePrice"].median())
def remove_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[col] >= lower) & (data[col] <= upper)]

num_cols_all = df.select_dtypes(include=["int64", "float64"]).columns
num_cols_all = num_cols_all.drop("SalePrice")

for col in num_cols_all:
    df = remove_outliers_iqr(df, col)
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].astype(str)
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore",drop="first")),
    ]), cat_cols)
])

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    pipe,
    X,
    y,
    cv=kf,
    scoring="r2",
    n_jobs=-1
)
print("Mean R2:", cv_scores.mean())
print("Std Dev:", cv_scores.std())