import pandas as pd
import os

# Step 1: Locate project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Step 2: Define data path
DATA_PATH = os.path.join(BASE_DIR, "data", "credit_data.csv")

# Step 3: Load dataset
data = pd.read_csv(DATA_PATH)

print("✅ Task 1: Credit Risk Data Loaded Successfully")

# Step 4: Basic inspection
print("\nDataset Shape (rows, columns):")
print(data.shape)

print("\nColumn Names:")
print(data.columns.tolist())

print("\nFirst 5 rows of the dataset:")
print(data.head())

print("\nData Types:")
print(data.dtypes)
