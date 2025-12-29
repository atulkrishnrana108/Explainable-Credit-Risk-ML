import pandas as pd
import os
import shap
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Step 1: Locate project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Step 2: Load data
X_train = pd.read_csv(os.path.join(BASE_DIR, "data", "X_train.csv"))
X_test  = pd.read_csv(os.path.join(BASE_DIR, "data", "X_test.csv"))
y_train = pd.read_csv(os.path.join(BASE_DIR, "data", "y_train.csv")).values.ravel()

print("✅ Task 13: SHAP Local Explainability Started")

# Step 3: Train XGBoost model
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

# Step 4: SHAP explainer
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Step 5: Select one instance (e.g., first test sample)
instance_index = 0

# Step 6: Waterfall plot (local explanation)
shap.plots.waterfall(shap_values[instance_index], show=True)

print("✅ Task 13 completed successfully!")
