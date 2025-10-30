# train_models.py
"""
Train ML Models for Waterborne Disease Prediction
With detailed performance metrics
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

print("\n" + "=" * 80)
print("MODEL TRAINING - WATERBORNE DISEASE PREDICTION")
print("Northeast India Community Health Monitoring")
print("=" * 80)

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Load data
print("\n[1/4] Loading dataset...")
df = pd.read_csv("data/raw/water_quality_data.csv")
print(f"✓ Loaded {len(df):,} records")
print(f"✓ Date range: {df['sample_date'].min()} to {df['sample_date'].max()}")

# Encode categorical variables
print("\n[2/4] Preprocessing data...")
label_encoders = {}
categorical_cols = ["state", "location_type", "water_source", "season"]

for col in categorical_cols:
    le = LabelEncoder()
    df[f"{col}_encoded"] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save encoders
with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print(f"✓ Encoded {len(categorical_cols)} categorical features")

# Select features
feature_cols = [
    "state_encoded",
    "location_type_encoded",
    "water_source_encoded",
    "season_encoded",
    "ph",
    "turbidity_ntu",
    "tds_mg_l",
    "dissolved_oxygen_mg_l",
    "bod_mg_l",
    "fecal_coliform_mpn",
    "total_coliform_mpn",
    "nitrate_mg_l",
    "fluoride_mg_l",
    "chloride_mg_l",
    "hardness_mg_l",
    "temperature_c",
    "arsenic_ug_l",
    "iron_mg_l",
    "population_served",
    "sanitation_access_percent",
]

X = df[feature_cols]
print(f"✓ Selected {len(feature_cols)} features")

# Train models
print("\n[3/4] Training disease-specific models...")
print("This will take 2-3 minutes...\n")

diseases = {
    "cholera": "cholera_outbreak",
    "typhoid": "typhoid_outbreak",
    "dysentery": "dysentery_outbreak",
    "hepatitis_a": "hepatitis_a_outbreak",
    "overall": "overall_outbreak",
}

disease_models = {}
disease_scalers = {}
disease_results = []

for idx, (disease_name, target_col) in enumerate(diseases.items(), 1):
    print(f"[{idx}/5] Training: {disease_name.replace('_', ' ').title()}")

    y = df[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"    Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    print(f"    SMOTE: {len(X_train_balanced):,} samples")

    # Train model
    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(X_train_balanced, y_train_balanced)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train_balanced, y_train_balanced, cv=5, scoring="accuracy", n_jobs=-1
    )

    print(
        f"    Accuracy: {accuracy * 100:>6.2f}% | Precision: {precision * 100:>6.2f}% | Recall: {recall * 100:>6.2f}% | F1: {f1 * 100:>6.2f}% | AUC: {roc_auc:.3f}"
    )
    print(
        f"    CV Accuracy: {cv_scores.mean() * 100:>6.2f}% ± {cv_scores.std() * 100:>4.2f}%\n"
    )

    # Save model and scaler
    disease_models[disease_name] = model
    disease_scalers[disease_name] = scaler

    disease_results.append(
        {
            "Disease": disease_name.replace("_", " ").title(),
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "CV_Accuracy_Mean": round(cv_scores.mean(), 4),
            "CV_Accuracy_Std": round(cv_scores.std(), 4),
        }
    )

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Outbreak", "Outbreak"],
        yticklabels=["No Outbreak", "Outbreak"],
    )
    plt.title(
        f"{disease_name.replace('_', ' ').title()} - Confusion Matrix",
        fontweight="bold",
        fontsize=12,
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(
        f"reports/{disease_name}_confusion_matrix.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

# Save everything
print("=" * 80)
print("[4/4] Saving models and metadata")
print("=" * 80 + "\n")

# Save models
for disease_name, model in disease_models.items():
    with open(f"models/{disease_name}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"  ✓ Saved: {disease_name}_model.pkl")

# Save scalers
for disease_name, scaler in disease_scalers.items():
    with open(f"models/{disease_name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

# Save feature names
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

# Save metadata with all results
metadata = {
    "project": "Water-Borne Disease Early Warning System",
    "region": "Northeast India",
    "dataset_size": int(len(df)),
    "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "diseases": list(diseases.keys()),
    "features": feature_cols,
    "categorical_features": categorical_cols,
    "model_type": "XGBoost Classifier",
    "results": disease_results,
}

with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

# Save results CSV
results_df = pd.DataFrame(disease_results)
results_df.to_csv("reports/model_performance.csv", index=False)

# Feature importance
overall_model = disease_models["overall"]
feature_importance = pd.DataFrame(
    {"Feature": feature_cols, "Importance": overall_model.feature_importances_}
).sort_values("Importance", ascending=False)

feature_importance.to_csv("reports/feature_importance.csv", index=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
plt.barh(top_features["Feature"], top_features["Importance"], color=colors)
plt.xlabel("Importance Score", fontweight="bold")
plt.ylabel("Feature", fontweight="bold")
plt.title(
    "Top 15 Most Important Features for Disease Prediction",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("reports/feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\n  ✓ Metadata: metadata.json")
print(f"  ✓ Results: model_performance.csv")
print(f"  ✓ Feature importance: feature_importance.csv")
print(f"  ✓ Visualizations: {len(diseases)} confusion matrices + feature importance")

# Final summary
print("\n" + "=" * 80)
print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)

print(f"\nDataset Size:     {len(df):,} water quality records")
print(f"Models Trained:   {len(diseases)}")
print(f"Features Used:    {len(feature_cols)}")
print(f"Model Type:       XGBoost Classifier")

print("\nModel Performance Summary:")
print("-" * 110)
print(
    f"{'Disease':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC':<10} {'CV Acc':<12}"
)
print("-" * 110)
for result in disease_results:
    print(
        f"{result['Disease']:<15} "
        f"{result['Accuracy'] * 100:>6.2f}%     "
        f"{result['Precision'] * 100:>6.2f}%     "
        f"{result['Recall'] * 100:>6.2f}%     "
        f"{result['F1-Score'] * 100:>6.2f}%     "
        f"{result['ROC-AUC']:>6.3f}    "
        f"{result['CV_Accuracy_Mean'] * 100:>6.2f}%"
    )
print("-" * 110)

avg_accuracy = sum(r["Accuracy"] for r in disease_results) / len(disease_results)
print(f"\nAverage Accuracy: {avg_accuracy * 100:.2f}%")

print("\nTop 10 Most Important Features:")
print("-" * 50)
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {idx + 1:2d}. {row['Feature']:30s}: {row['Importance']:.4f}")

print("\n" + "=" * 80)
print("NEXT STEP: Run the Streamlit Dashboard")
print("=" * 80)
print("\nCommand: streamlit run app.py")
print("\nThe system is now ready to monitor water quality and predict outbreaks!")
print("=" * 80 + "\n")
