import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib # For saving the model
import os

DATA_DIR = "gesture_data"
gestures = ["FIST", "OPEN_PALM", "V_SIGN", "ILY_SIGN"] # Must match your data_collector gestures

all_data = []
for gesture_name in gestures:
    csv_file_path = os.path.join(DATA_DIR, f"{gesture_name}.csv")
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
        all_data.append(df)
    else:
        print(f"Warning: CSV file not found for {gesture_name}: {csv_file_path}")

if not all_data:
    print("No data found to train the model. Please run data_collector.py first.")
    exit()

combined_df = pd.concat(all_data, ignore_index=True)

# Separate features (X) and labels (y)
X = combined_df.drop("label", axis=1)
y = combined_df["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train a RandomForestClassifier
# You can experiment with other models like SVC, K-Nearest Neighbors, or even a simple Neural Network
model = RandomForestClassifier(n_estimators=100, random_state=42)
print("Training model...")
model.fit(X_train, y_train)
print("Model training complete.")

# Evaluate the model
y_pred = model.predict(X_test)
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
model_save_path = "gesture_recognition_model.joblib"
joblib.dump(model, model_save_path)
print(f"\nModel saved to {model_save_path}")