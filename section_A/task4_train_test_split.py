import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Step 1: Locate project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Step 2: Load scaled dataset
DATA_PATH = os.path.join(BASE_DIR, "data", "credit_data_scaled.csv")
df = pd.read_csv(DATA_PATH)

print("✅ Task 4: Train-Test Split Started")

# Step 3: Separate features and target
X = df.drop("default", axis=1)
y = df["default"]

# Step 4: Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 5: Display split sizes
print("\n📊 Dataset split summary:")
print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# Step 6: Save split datasets
X_train.to_csv(os.path.join(BASE_DIR, "data", "X_train.csv"), index=False)
X_test.to_csv(os.path.join(BASE_DIR, "data", "X_test.csv"), index=False)
y_train.to_csv(os.path.join(BASE_DIR, "data", "y_train.csv"), index=False)
y_test.to_csv(os.path.join(BASE_DIR, "data", "y_test.csv"), index=False)

print("\n💾 Train-test datasets saved successfully")
print("✅ Task 4 completed successfully!")
