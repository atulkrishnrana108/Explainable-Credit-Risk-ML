import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# Step 1: Locate project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Step 2: Load cleaned dataset
DATA_PATH = os.path.join(BASE_DIR, "data", "credit_data_cleaned.csv")
df = pd.read_csv(DATA_PATH)

print("✅ Task 3: Feature Scaling Started")

# Step 3: Separate features and target
X = df.drop("default", axis=1)
y = df["default"]

print("\n📌 Features before scaling:")
print(X.head())

# Step 4: Apply Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Convert back to DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("\n📊 Features after scaling:")
print(X_scaled_df.head())

# Step 6: Combine scaled features with target
scaled_df = pd.concat([X_scaled_df, y], axis=1)

# Step 7: Save normalized dataset
SCALED_PATH = os.path.join(BASE_DIR, "data", "credit_data_scaled.csv")
scaled_df.to_csv(SCALED_PATH, index=False)

print("\n💾 Scaled dataset saved as credit_data_scaled.csv")
print("✅ Task 3 completed successfully!")
