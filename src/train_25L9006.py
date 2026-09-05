import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

# 1. Load dataset from data/ directory
print("Loading dataset...")
data_path = os.path.join("data", "dataset.csv")
df = pd.read_csv(data_path)

# Drop non-numeric / irrelevant columns (id, date, etc.)
df = df.drop(columns=["id", "date"], errors="ignore")

# Drop any remaining non-numeric columns just in case
df = df.select_dtypes(include=["number"])

# Assume last column is the target (price)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 2. Train a machine learning model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = (X_train - X_train.mean()) / X_train.std()  # normalization step

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Model trained. R^2 Score: {score:.2f}")

# 3. Save the trained model into model/ directory
os.makedirs("model", exist_ok=True)
model_path = os.path.join("model", "trained_model.pkl")
joblib.dump(model, model_path)

print(f"Model saved to {model_path}")