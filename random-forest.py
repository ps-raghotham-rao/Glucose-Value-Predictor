#random forest regressor

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
df["insulin_ratio"] = df["Insulin"] / df["Glucose"]

# Split into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=42)

# Normalize/Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Perform PCA for dimensionality reduction
pca = PCA(n_components=5)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Train the model
model = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42)
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


'''
Note that in this program, we used a random forest regressor with 100 trees and a maximum depth of 10 to prevent overfitting. You may need to experiment with different hyperparameters and algorithms to find the best model for your specific problem.
'''