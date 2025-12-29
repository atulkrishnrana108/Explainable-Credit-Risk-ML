import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Locate project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Step 2: Load train-test data
X_train = pd.read_csv(os.path.join(BASE_DIR, "data", "X_train.csv"))
X_test  = pd.read_csv(os.path.join(BASE_DIR, "data", "X_test.csv"))
y_train = pd.read_csv(os.path.join(BASE_DIR, "data", "y_train.csv")).values.ravel()
y_test  = pd.read_csv(os.path.join(BASE_DIR, "data", "y_test.csv")).values.ravel()

print("✅ Task 7: Random Forest Training Started")

# Step 3: Initialize Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Step 4: Train model
rf_model.fit(X_train, y_train)
print("✅ Random Forest model trained")

# Step 5: Predict
y_pred = rf_model.predict(X_test)

# Step 6: Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("\n📊 Random Forest Accuracy:", accuracy)

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

print("✅ Task 7 completed successfully!")
