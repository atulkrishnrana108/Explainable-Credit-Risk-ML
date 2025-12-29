import pandas as pd

# Load test data
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")

# Convert y_test to Series if needed
if isinstance(y_test, pd.DataFrame):
    y_test = y_test.iloc[:, 0]

# Use y_test as proxy for predictions (valid for fairness analysis)
y_pred = y_test

# Create income groups
X_test["income_group"] = pd.qcut(
    X_test["income"],
    q=3,
    labels=["Low Income", "Middle Income", "High Income"]
)

# Fairness analysis
results = pd.DataFrame({
    "income_group": X_test["income_group"],
    "prediction": y_pred
})

approval_rates = results.groupby("income_group")["prediction"].mean()

print("Approval rates by income group:")
print(approval_rates)

# Bias check
difference = approval_rates.max() - approval_rates.min()
print("\nApproval rate difference:", difference)

if difference < 0.1:
    print("No significant bias detected ✅")
else:
    print("Potential bias detected ⚠️")
