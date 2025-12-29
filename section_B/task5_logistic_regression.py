import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Locate project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Step 2: Load train-test data
X_train = pd.read_csv(os.path.join(BASE_DIR, "data", "X_train.csv"))
X_test  = pd.read_csv(os.path.join(BASE_DIR, "data", "X_test.csv"))
y_train = pd.read_csv(os.path.join(BASE_DIR, "data", "y_train.csv"))
y_test  = pd.read_csv(os.path.join(BASE_DIR, "data", "y_test.csv"))

# Convert target to 1D array
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print("✅ Task 5: Logistic Regression Started")

# Step 3: Initialize model
model = LogisticRegression(max_iter=1000)

# Step 4: Train model
model.fit(X_train, y_train)
print("✅ Logistic Regression model trained")

# Step 5: Predict on test data
y_pred = model.predict(X_test)

# Step 6: Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("\n📊 Model Accuracy:", accuracy)

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

print("✅ Task 5 completed successfully!")
