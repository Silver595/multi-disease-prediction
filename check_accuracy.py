# check_accuracy.py
"""
Check Model Accuracy - Detailed Performance Report
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "=" * 80)
print("MODEL ACCURACY & PERFORMANCE CHECKER")
print("=" * 80)

# Load model and data
print("\n[1/4] Loading model and data...")
try:
    with open("models/disease_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    with open("models/model_metadata.json", "r") as f:
        metadata = json.load(f)
    print("✓ Model files loaded successfully")
except FileNotFoundError as e:
    print(f"❌ Error: Model files not found. Run 'python create_and_train.py' first.")
    exit(1)

# Load dataset
df = pd.read_csv("data/health_dataset.csv")
print(f"✓ Dataset loaded: {len(df)} records")

# Prepare test data
X = df.drop("disease_risk", axis=1)
y = df["disease_risk"]

# Split same way as training
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
X_test_scaled = scaler.transform(X_test)

# Make predictions
print("\n[2/4] Making predictions on test set...")
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
print("✓ Predictions complete")

# Calculate all metrics
print("\n[3/4] Calculating performance metrics...")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Additional metrics
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
positive_predictive_value = tp / (tp + fp)
negative_predictive_value = tn / (tn + fn)

print("✓ Metrics calculated")

# Display Results
print("\n" + "=" * 80)
print("📊 MODEL PERFORMANCE METRICS")
print("=" * 80)

print(f"\n{'Metric':<30} {'Value':<15} {'Percentage':<15}")
print("-" * 60)
print(f"{'Accuracy':<30} {accuracy:<15.4f} {accuracy * 100:>6.2f}%")
print(f"{'Precision':<30} {precision:<15.4f} {precision * 100:>6.2f}%")
print(f"{'Recall (Sensitivity)':<30} {recall:<15.4f} {recall * 100:>6.2f}%")
print(f"{'Specificity':<30} {specificity:<15.4f} {specificity * 100:>6.2f}%")
print(f"{'F1-Score':<30} {f1:<15.4f} {f1 * 100:>6.2f}%")
print(f"{'ROC-AUC Score':<30} {roc_auc:<15.4f} {roc_auc * 100:>6.2f}%")
print(
    f"{'PPV (Precision)':<30} {positive_predictive_value:<15.4f} {positive_predictive_value * 100:>6.2f}%"
)
print(
    f"{'NPV':<30} {negative_predictive_value:<15.4f} {negative_predictive_value * 100:>6.2f}%"
)

print("\n" + "=" * 80)
print("📊 CONFUSION MATRIX")
print("=" * 80)
print(f"\n                    Predicted")
print(f"                Low Risk    High Risk")
print(f"Actual  Low Risk    {tn:6d}       {fp:6d}")
print(f"       High Risk    {fn:6d}       {tp:6d}")

print("\n" + "=" * 80)
print("📊 CONFUSION MATRIX BREAKDOWN")
print("=" * 80)
print(f"True Negatives (Correct Low Risk):   {tn:6d}")
print(f"True Positives (Correct High Risk):  {tp:6d}")
print(f"False Positives (Wrong High Risk):   {fp:6d}")
print(f"False Negatives (Missed High Risk):  {fn:6d}")

print("\n" + "=" * 80)
print("📊 MODEL PERFORMANCE INTERPRETATION")
print("=" * 80)

if accuracy >= 0.95:
    grade = "EXCELLENT ⭐⭐⭐⭐⭐"
elif accuracy >= 0.90:
    grade = "VERY GOOD ⭐⭐⭐⭐"
elif accuracy >= 0.85:
    grade = "GOOD ⭐⭐⭐"
elif accuracy >= 0.80:
    grade = "FAIR ⭐⭐"
else:
    grade = "NEEDS IMPROVEMENT ⭐"

print(f"\nOverall Model Grade: {grade}")
print(f"\nThe model correctly predicts {accuracy * 100:.2f}% of all cases.")
print(f"Out of 100 predictions, approximately {int(accuracy * 100)} will be correct.")

# Detailed report
print("\n" + "=" * 80)
print("📊 DETAILED CLASSIFICATION REPORT")
print("=" * 80)
print(
    classification_report(
        y_test, y_pred, target_names=["Low Risk", "High Risk"], digits=4
    )
)

# Create visualizations
print("\n[4/4] Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Confusion Matrix
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[0, 0],
    xticklabels=["Low Risk", "High Risk"],
    yticklabels=["Low Risk", "High Risk"],
)
axes[0, 0].set_title("Confusion Matrix", fontsize=14, fontweight="bold")
axes[0, 0].set_ylabel("Actual")
axes[0, 0].set_xlabel("Predicted")

# 2. Metrics Bar Chart
metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
metrics_values = [accuracy, precision, recall, f1, roc_auc]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

bars = axes[0, 1].barh(metrics_names, metrics_values, color=colors)
axes[0, 1].set_xlim([0, 1])
axes[0, 1].set_xlabel("Score")
axes[0, 1].set_title("Model Performance Metrics", fontsize=14, fontweight="bold")
axes[0, 1].grid(axis="x", alpha=0.3)

# Add value labels
for i, (bar, value) in enumerate(zip(bars, metrics_values)):
    axes[0, 1].text(
        value + 0.01, i, f"{value:.4f} ({value * 100:.2f}%)", va="center", fontsize=10
    )

# 3. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
axes[1, 0].plot(
    fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
)
axes[1, 0].plot(
    [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier"
)
axes[1, 0].set_xlim([0.0, 1.0])
axes[1, 0].set_ylim([0.0, 1.05])
axes[1, 0].set_xlabel("False Positive Rate")
axes[1, 0].set_ylabel("True Positive Rate")
axes[1, 0].set_title("ROC Curve", fontsize=14, fontweight="bold")
axes[1, 0].legend(loc="lower right")
axes[1, 0].grid(alpha=0.3)

# 4. Prediction Distribution
axes[1, 1].hist(
    [y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]],
    bins=30,
    label=["Actual Low Risk", "Actual High Risk"],
    color=["green", "red"],
    alpha=0.7,
)
axes[1, 1].axvline(
    x=0.5, color="black", linestyle="--", linewidth=2, label="Decision Threshold"
)
axes[1, 1].set_xlabel("Predicted Probability")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].set_title(
    "Prediction Probability Distribution", fontsize=14, fontweight="bold"
)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("reports/model_accuracy_report.png", dpi=300, bbox_inches="tight")
print("✓ Visualization saved: reports/model_accuracy_report.png")

# Save detailed report to file
print("\n[5/5] Saving detailed report...")

report_text = f"""
================================================================================
MODEL ACCURACY & PERFORMANCE REPORT
================================================================================

Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
Model: {metadata.get("model_name", "Unknown")}
Training Date: {metadata.get("training_date", "Unknown")}

================================================================================
PERFORMANCE METRICS
================================================================================

Accuracy:           {accuracy:.4f} ({accuracy * 100:.2f}%)
Precision:          {precision:.4f} ({precision * 100:.2f}%)
Recall:             {recall:.4f} ({recall * 100:.2f}%)
Specificity:        {specificity:.4f} ({specificity * 100:.2f}%)
F1-Score:           {f1:.4f} ({f1 * 100:.2f}%)
ROC-AUC Score:      {roc_auc:.4f} ({roc_auc * 100:.2f}%)

================================================================================
CONFUSION MATRIX
================================================================================

                    Predicted
                Low Risk    High Risk
Actual  Low Risk    {tn:6d}       {fp:6d}
       High Risk    {fn:6d}       {tp:6d}

True Negatives:  {tn}
True Positives:  {tp}
False Positives: {fp}
False Negatives: {fn}

================================================================================
MODEL GRADE: {grade}
================================================================================

The model correctly predicts {accuracy * 100:.2f}% of all cases.

================================================================================
INTERPRETATION
================================================================================

• Accuracy: {accuracy * 100:.2f}% of all predictions are correct
• Precision: {precision * 100:.2f}% of high-risk predictions are actually high-risk
• Recall: {recall * 100:.2f}% of actual high-risk cases are correctly identified
• Specificity: {specificity * 100:.2f}% of actual low-risk cases are correctly identified

================================================================================
"""

with open("reports/accuracy_report.txt", "w") as f:
    f.write(report_text)

print("✓ Report saved: reports/accuracy_report.txt")

print("\n" + "=" * 80)
print("✅ ACCURACY CHECK COMPLETED!")
print("=" * 80)
print(f"\n🎯 MODEL ACCURACY: {accuracy * 100:.2f}%")
print(f"📊 Grade: {grade}")
print("\nFiles created:")
print("  • reports/model_accuracy_report.png")
print("  • reports/accuracy_report.txt")
print("\nRun 'streamlit run app.py' to see accuracy in the web interface!")
print("=" * 80 + "\n")
