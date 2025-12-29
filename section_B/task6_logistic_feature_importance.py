import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Step 1: Locate project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Step 2: Load train data
X_train = pd.read_csv(os.path.join(BASE_DIR, "data", "X_train.csv"))
y_train = pd.read_csv(os.path.join(BASE_DIR, "data", "y_train.csv")).values.ravel()

print("✅ Task 6: Logistic Regression Feature Importance Started")

# Step 3: Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 4: Extract coefficients
coefficients = model.coef_[0]

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", ascending=False)

print("\n📊 Feature Importance (Logistic Regression):")
print(feature_importance)

# Step 5: Plot feature importance
plt.figure(figsize=(8, 5))
plt.barh(feature_importance["Feature"], feature_importance["Coefficient"])
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title("Logistic Regression Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("✅ Task 6 completed successfully!")
