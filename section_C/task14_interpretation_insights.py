import os

# Step 1: Locate project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("✅ Task 14: Interpretation & Insights Started")

# Step 2: Define insights text
insights = """
INTERPRETATION & INSIGHTS – CREDIT RISK PREDICTION

Global Explainability (SHAP Summary):
The SHAP summary analysis indicates that loan_amount is the most influential
feature increasing the probability of loan default. Applicants requesting
higher loan amounts are associated with higher default risk.

Credit_history and income have strong negative SHAP values, indicating that
a good credit history and higher income significantly reduce default risk.
Employment_years also contributes to stability by lowering default probability.

Local Explainability (SHAP Waterfall):
The SHAP waterfall plot for an individual applicant demonstrates how specific
features collectively influence the model’s decision. In high-risk cases,
large loan amounts push predictions toward default, while good credit history
and sufficient income counterbalance this risk.

Model Behavior and Trust:
By combining global and local explainability, the model ensures transparency
and trustworthiness. Stakeholders can understand both overall trends and
individual decisions, which is critical in regulated domains such as finance.

Conclusion:
The explainable credit risk framework successfully balances predictive
performance with interpretability, making it suitable for responsible and
transparent decision-making in real-world credit assessment.
"""

# Step 3: Save insights to file
OUTPUT_PATH = os.path.join(BASE_DIR, "section_C", "model_interpretation.txt")
with open(OUTPUT_PATH, "w") as f:
    f.write(insights)

print("💾 Interpretation insights saved to section_C/model_interpretation.txt")
print("✅ Task 14 completed successfully!")
