# create_and_train.py
"""
Enhanced Disease Risk Prediction System
Now predicts specific diseases: Diabetes, Heart Disease, Hypertension, Stroke
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

print("\n" + "=" * 80)
print("DISEASE RISK PREDICTION SYSTEM - COMPLETE SETUP")
print("Predicting: Diabetes, Heart Disease, Hypertension, Stroke")
print("=" * 80)

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ============================================================================
# STEP 1: Generate Realistic Dataset with Specific Diseases
# ============================================================================
print("\n[1/4] Generating Dataset with Disease-Specific Labels...")

np.random.seed(42)
n_samples = 5000

# Generate base demographics
age = np.random.randint(20, 85, n_samples)
gender = np.random.randint(0, 2, n_samples)
bmi_base = 22 + (age - 40) * 0.08 + np.random.normal(0, 4, n_samples)
bmi = np.clip(bmi_base, 16, 45)

# Vitals
bp_base = 100 + (age * 0.4) + (bmi - 22) * 1.2 + np.random.normal(0, 12, n_samples)
blood_pressure = np.clip(bp_base, 90, 200).astype(int)

chol_base = 150 + (age * 0.8) + (bmi - 22) * 2.0 + np.random.normal(0, 25, n_samples)
cholesterol = np.clip(chol_base, 120, 350).astype(int)

family_history = np.random.binomial(1, 0.35, n_samples)
sugar_base = (
    80
    + (age * 0.3)
    + (bmi - 22) * 1.5
    + (family_history * 15)
    + np.random.normal(0, 18, n_samples)
)
blood_sugar = np.clip(sugar_base, 70, 250).astype(int)

heart_rate = np.clip(
    75 - (age - 40) * 0.1 + np.random.normal(0, 10, n_samples), 50, 120
).astype(int)
smoking = np.random.binomial(1, 0.22, n_samples)
exercise_base = (
    8 - (bmi - 22) * 0.2 - (age - 40) * 0.05 + np.random.normal(0, 2.5, n_samples)
)
exercise_hours = np.clip(exercise_base, 0, 20)

df = pd.DataFrame(
    {
        "age": age,
        "gender": gender,
        "bmi": np.round(bmi, 1),
        "blood_pressure": blood_pressure,
        "cholesterol": cholesterol,
        "blood_sugar": blood_sugar,
        "heart_rate": heart_rate,
        "smoking": smoking,
        "exercise_hours": np.round(exercise_hours, 1),
        "family_history": family_history,
    }
)

# Create disease-specific risk scores
print("  Calculating disease-specific risks...")

# DIABETES
diabetes_score = (
    (df["blood_sugar"] > 126) * 50
    + (df["blood_sugar"] > 100) * 25
    + (df["bmi"] > 30) * 30
    + df["family_history"] * 35
    + (df["age"] > 45) * 20
    + (df["exercise_hours"] < 3) * 15
    + np.random.normal(0, 10, n_samples)
)
df["diabetes"] = (diabetes_score > 80).astype(int)

# HEART DISEASE
heart_score = (
    (df["cholesterol"] > 240) * 40
    + (df["blood_pressure"] > 140) * 35
    + df["smoking"] * 40
    + (df["age"] > 55) * 30
    + (df["bmi"] > 30) * 25
    + df["family_history"] * 30
    + np.random.normal(0, 12, n_samples)
)
df["heart_disease"] = (heart_score > 90).astype(int)

# HYPERTENSION
hypertension_score = (
    (df["blood_pressure"] > 160) * 60
    + (df["blood_pressure"] > 140) * 40
    + (df["bmi"] > 30) * 30
    + (df["age"] > 50) * 25
    + df["smoking"] * 20
    + df["family_history"] * 25
    + np.random.normal(0, 10, n_samples)
)
df["hypertension"] = (hypertension_score > 85).astype(int)

# STROKE
stroke_score = (
    (df["blood_pressure"] > 160) * 45
    + (df["age"] > 65) * 40
    + df["smoking"] * 35
    + (df["cholesterol"] > 240) * 25
    + (df["blood_sugar"] > 126) * 20
    + df["family_history"] * 30
    + np.random.normal(0, 12, n_samples)
)
df["stroke"] = (stroke_score > 95).astype(int)

# Overall disease risk (if any disease is high risk)
df["disease_risk"] = (
    (df["diabetes"] == 1)
    | (df["heart_disease"] == 1)
    | (df["hypertension"] == 1)
    | (df["stroke"] == 1)
).astype(int)

# Save dataset
df.to_csv("data/health_dataset.csv", index=False)

print(f"\n✓ Generated {len(df)} records")
print(f"\nDisease Distribution:")
print(
    f"  Diabetes:      {df['diabetes'].sum():4d} cases ({df['diabetes'].mean() * 100:.1f}%)"
)
print(
    f"  Heart Disease: {df['heart_disease'].sum():4d} cases ({df['heart_disease'].mean() * 100:.1f}%)"
)
print(
    f"  Hypertension:  {df['hypertension'].sum():4d} cases ({df['hypertension'].mean() * 100:.1f}%)"
)
print(
    f"  Stroke:        {df['stroke'].sum():4d} cases ({df['stroke'].mean() * 100:.1f}%)"
)
print(
    f"  Overall Risk:  {df['disease_risk'].sum():4d} cases ({df['disease_risk'].mean() * 100:.1f}%)"
)

# ============================================================================
# STEP 2: Train Individual Models for Each Disease
# ============================================================================
print("\n[2/4] Training Disease-Specific Models...")

X = df[
    [
        "age",
        "gender",
        "bmi",
        "blood_pressure",
        "cholesterol",
        "blood_sugar",
        "heart_rate",
        "smoking",
        "exercise_hours",
        "family_history",
    ]
]
feature_names = X.columns.tolist()

diseases = ["diabetes", "heart_disease", "hypertension", "stroke", "disease_risk"]
disease_models = {}
disease_scalers = {}
disease_results = []

for disease in diseases:
    print(f"\n{'─' * 80}")
    print(f"Training: {disease.replace('_', ' ').title()}")
    print(f"{'─' * 80}")

    y = df[disease]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )

    model.fit(X_train_balanced, y_train_balanced)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"  Accuracy:  {accuracy * 100:.2f}%")
    print(f"  Precision: {precision * 100:.2f}%")
    print(f"  Recall:    {recall * 100:.2f}%")
    print(f"  F1-Score:  {f1 * 100:.2f}%")

    disease_models[disease] = model
    disease_scalers[disease] = scaler
    disease_results.append(
        {
            "Disease": disease.replace("_", " ").title(),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
        }
    )

# ============================================================================
# STEP 3: Save All Models
# ============================================================================
print("\n[3/4] Saving Models...")

# Save disease-specific models
for disease, model in disease_models.items():
    with open(f"models/{disease}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"✓ Saved: models/{disease}_model.pkl")

# Save scalers
for disease, scaler in disease_scalers.items():
    with open(f"models/{disease}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

# Save feature names
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)
print(f"✓ Saved: models/feature_names.pkl")

# Save metadata
import json

metadata = {
    "diseases": diseases,
    "features": feature_names,
    "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "training_samples": len(X_train_balanced),
    "results": disease_results,
}

with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)
print(f"✓ Saved: models/model_metadata.json")

# Save sample test data
sample_df = df.drop(
    ["diabetes", "heart_disease", "hypertension", "stroke", "disease_risk"], axis=1
).sample(n=20, random_state=42)
sample_df.to_csv("data/sample_test.csv", index=False)
print(f"✓ Saved: data/sample_test.csv")

# ============================================================================
# STEP 4: Create Summary Report
# ============================================================================
print("\n[4/4] Creating Reports...")

results_df = pd.DataFrame(disease_results)

print("\n" + "=" * 80)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 80)
print(results_df.to_string(index=False))

# Save to CSV
results_df.to_csv("reports/model_comparison.csv", index=False)
print(f"\n✓ Saved: reports/model_comparison.csv")

# Create visualization
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(results_df))
width = 0.2

ax.bar(x - width * 1.5, results_df["Accuracy"], width, label="Accuracy", alpha=0.8)
ax.bar(x - width * 0.5, results_df["Precision"], width, label="Precision", alpha=0.8)
ax.bar(x + width * 0.5, results_df["Recall"], width, label="Recall", alpha=0.8)
ax.bar(x + width * 1.5, results_df["F1-Score"], width, label="F1-Score", alpha=0.8)

ax.set_ylabel("Score", fontweight="bold")
ax.set_title("Model Performance by Disease", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(results_df["Disease"], rotation=15, ha="right")
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("reports/model_performance.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: reports/model_performance.png")

print("\n" + "=" * 80)
print("✅ SETUP COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\nFiles Created:")
print(f"  • data/health_dataset.csv ({len(df)} records)")
print(f"  • models/*_model.pkl (5 disease models)")
print(f"  • models/*_scaler.pkl (5 scalers)")
print(f"  • models/feature_names.pkl")
print(f"  • models/model_metadata.json")
print(f"  • reports/model_comparison.csv")
print(f"  • reports/model_performance.png")
print("\n" + "=" * 80)
print("NEXT STEP: Run the Streamlit application")
print("Command: streamlit run app.py")
print("=" * 80 + "\n")
