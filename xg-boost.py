# make sure to pip install xgboost


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("output.csv")

# Remove outliers using the IQR method
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Create new features
df["bmi_skin"] = df["BMI"] * df["SkinThickness"]
# df["insulin_ratio"] = df["Insulin"] / df["Glucose"]

# Split into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=42)

# Normalize/Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Perform feature selection
selector = SelectKBest(f_regression, k='all')
selector.fit(X_train, y_train)
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)

# Perform PCA for dimensionality reduction
pca = PCA(n_components=5)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Train an XGBoost model using randomized search to optimize hyperparameters
model = XGBRegressor(random_state=42)
params = {
    'n_estimators': [100, 500],
    'max_depth': [10, 15],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.5, 0.8],
    'colsample_bytree': [0.5, 0.8],
    'reg_lambda': [1, 10],
    'reg_alpha': [0, 0.1]
}
model = RandomizedSearchCV(model, params, n_iter=10, cv=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-Squared:", r2)

# Visualize the model's predictions
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Glucose Value")
plt.ylabel("Predicted Glucose Value")
plt.title("Random Forest Regressor Predictions")

plt.show()
