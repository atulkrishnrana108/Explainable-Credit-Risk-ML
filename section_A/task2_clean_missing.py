import pandas as pd
import os

# Step 1: Locate project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Step 2: Load dataset
DATA_PATH = os.path.join(BASE_DIR, "data", "credit_data.csv")
df = pd.read_csv(DATA_PATH)

print("✅ Task 2: Data Cleaning Started")

# Step 3: Check missing values
print("\n🔍 Missing values per column:")
print(df.isnull().sum())

# Step 4: Handle missing values
# Numeric columns → fill with median (robust to outliers)
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Step 5: Validate data types
print("\n📊 Data types after cleaning:")
print(df.dtypes)

# Step 6: Final missing value check
print("\n✅ Missing values after cleaning:")
print(df.isnull().sum())

# Step 7: Save cleaned dataset
CLEAN_PATH = os.path.join(BASE_DIR, "data", "credit_data_cleaned.csv")
df.to_csv(CLEAN_PATH, index=False)

print("\n💾 Cleaned dataset saved as credit_data_cleaned.csv")
print("✅ Task 2 completed successfully!")
