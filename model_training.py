# model_training.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

print("🚨 RUNNING NEW CODE VERSION 🚨")

# Load dataset
housing = fetch_california_housing(as_frame=True)
housing_df = housing.frame.copy()

print("✅ Columns in dataset:", housing_df.columns.tolist())

# Add only safe, valid features
housing_df["bedrooms_per_room"] = housing_df["AveBedrms"] / housing_df["AveRooms"]
housing_df["occupancy_per_room"] = housing_df["AveOccup"] / housing_df["AveRooms"]
housing_df["income_per_room"] = housing_df["MedInc"] / housing_df["AveRooms"]

# Split features and target
X = housing_df.drop("MedHouseVal", axis=1)
y = housing_df["MedHouseVal"]

# Impute missing values
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train models
lin_reg = LinearRegression().fit(X_train, y_train)
tree_reg = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
forest_reg = RandomForestRegressor(random_state=42).fit(X_train, y_train)

# Evaluation function
def evaluate(name, model):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}")

evaluate("Linear Regression", lin_reg)
evaluate("Decision Tree", tree_reg)
evaluate("Random Forest", forest_reg)

# Save best model
joblib.dump(forest_reg, "best_model.pkl")
print("✅ Model saved as 'best_model.pkl'")
