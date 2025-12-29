import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

# Step 1: Locate project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Step 2: Load train-test data
X_train = pd.read_csv(os.path.join(BASE_DIR, "data", "X_train.csv"))
X_test  = pd.read_csv(os.path.join(BASE_DIR, "data", "X_test.csv"))
y_train = pd.read_csv(os.path.join(BASE_DIR, "data", "y_train.csv")).values.ravel()
y_test  = pd.read_csv(os.path.join(BASE_DIR, "data", "y_test.csv")).values.ravel()

print("✅ Task 10: Confusion Matrix Generation Started")

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

# Step 4: Predict
y_pred = model.predict(X_test)

# Step 5: Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Step 6: Plot confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Default", "Default"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix – Credit Risk Prediction")
plt.tight_layout()
plt.show()

print("✅ Task 10 completed successfully!")
