import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("kota_flood_final_dataset.csv")

# Clean
data = data.dropna()

# -------------------------------
# 🔥 FEATURE ENGINEERING (NEW)
# -------------------------------

data["Rainfall_Intensity"] = data["Rainfall"] / (data["Rainfall_3Day"] + 1)
data["Rainfall_Change"] = data["Rainfall_3Day"] - data["Rainfall_7Day"]
data["Heavy_Rain"] = (data["Rainfall"] > 100).astype(int)
data["Accumulation_Ratio"] = data["Rainfall_3Day"] / (data["Rainfall_7Day"] + 1)

# Features & target
X = data[[
    "Rainfall",
    "Rainfall_3Day",
    "Rainfall_7Day",
    "Rainfall_Intensity",
    "Rainfall_Change",
    "Heavy_Rain",
    "Accumulation_Ratio"
]]

y = data["Flood"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 🌲 RANDOM FOREST MODEL
# -------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "flood_model.pkl")
print("Model saved successfully!")

# Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))