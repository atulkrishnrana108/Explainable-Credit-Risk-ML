import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Locate project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Step 2: Load train-test data
X_train = pd.read_csv(os.path.join(BASE_DIR, "data", "X_train.csv"))
X_test  = pd.read_csv(os.path.join(BASE_DIR, "data", "X_test.csv"))
y_train = pd.read_csv(os.path.join(BASE_DIR, "data", "y_train.csv")).values.ravel()
y_test  = pd.read_csv(os.path.join(BASE_DIR, "data", "y_test.csv")).values.ravel()

print("✅ Task 8: XGBoost Training Started")

# Step 3: Initialize XGBoost model
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

# Step 4: Train model
xgb_model.fit(X_train, y_train)
print("✅ XGBoost model trained")

# Step 5: Predict
y_pred = xgb_model.predict(X_test)

# Step 6: Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("\n📊 XGBoost Accuracy:", accuracy)

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

print("✅ Task 8 completed successfully!")
