import pandas as pd
import os
import shap
from xgboost import XGBClassifier

# Step 1: Locate project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Step 2: Load data
X_train = pd.read_csv(os.path.join(BASE_DIR, "data", "X_train.csv"))
X_test  = pd.read_csv(os.path.join(BASE_DIR, "data", "X_test.csv"))
y_train = pd.read_csv(os.path.join(BASE_DIR, "data", "y_train.csv")).values.ravel()

print("✅ Task 11: SHAP Setup Started")

# Step 3: Train best model (XGBoost)
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)
print("✅ XGBoost model trained for SHAP")

# Step 4: Initialize SHAP explainer
explainer = shap.Explainer(model, X_train)

# Step 5: Compute SHAP values
shap_values = explainer(X_test)

print("✅ SHAP values computed successfully")
print("✅ Task 11 completed!")
