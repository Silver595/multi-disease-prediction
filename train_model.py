# scripts/train_model.py
"""
Train disease risk prediction model from CSV dataset
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
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
from imblearn.over_sampling import SMOTE
import warnings
import os

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


def load_dataset(filepath):
    """Load dataset from CSV"""
    print("=" * 70)
    print(f"LOADING DATASET FROM: {filepath}")
    print("=" * 70)

    df = pd.read_csv(filepath)

    print(f"\n✓ Dataset loaded successfully!")
    print(f"✓ Shape: {df.shape}")
    print(f"✓ Columns: {len(df.columns)}")
    print(f"✓ Records: {len(df)}")

    return df


def explore_dataset(df):
    """Exploratory Data Analysis"""
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    print("\n1. Dataset Info:")
    print(df.info())

    print("\n2. Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✓ No missing values found!")
    else:
        print(missing[missing > 0])

    print("\n3. Duplicate Records:")
    duplicates = df.duplicated().sum()
    print(f"✓ Duplicates: {duplicates}")

    print("\n4. Target Distribution:")
    print(df["disease_risk"].value_counts())
    print("\nPercentage:")
    print(df["disease_risk"].value_counts(normalize=True) * 100)

    print("\n5. Statistical Summary:")
    print(df.describe())


def create_visualizations(df):
    """Generate and save visualizations"""
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    os.makedirs("visualizations", exist_ok=True)

    # Target distribution
    plt.figure(figsize=(10, 6))
    counts = df["disease_risk"].value_counts()
    plt.bar(["Low Risk", "High Risk"], counts.values, color=["green", "red"], alpha=0.7)
    plt.title("Disease Risk Distribution", fontsize=16, fontweight="bold")
    plt.ylabel("Number of Patients")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 50, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("visualizations/01_target_distribution.png", dpi=300)
    print("✓ Saved: 01_target_distribution.png")
    plt.close()

    # Correlation matrix
    plt.figure(figsize=(20, 16))
    correlation = df.corr()
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Matrix", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig("visualizations/02_correlation_matrix.png", dpi=300)
    print("✓ Saved: 02_correlation_matrix.png")
    plt.close()

    # Age distribution by risk
    plt.figure(figsize=(12, 6))
    plt.hist(
        [df[df["disease_risk"] == 0]["age"], df[df["disease_risk"] == 1]["age"]],
        bins=30,
        label=["Low Risk", "High Risk"],
        color=["green", "red"],
        alpha=0.7,
    )
    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Age Distribution by Risk Level", fontsize=16, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualizations/03_age_distribution.png", dpi=300)
    print("✓ Saved: 03_age_distribution.png")
    plt.close()


def train_and_compare_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance"""
    print("\n" + "=" * 70)
    print("TRAINING MULTIPLE MODELS")
    print("=" * 70)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, random_state=42, eval_metric="logloss", n_jobs=-1
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n{'─' * 70}")
        print(f"Training: {name}")
        print(f"{'─' * 70}")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        results.append(
            {
                "Model": name,
                "Accuracy": f"{accuracy:.4f}",
                "Precision": f"{precision:.4f}",
                "Recall": f"{recall:.4f}",
                "F1-Score": f"{f1:.4f}",
                "ROC-AUC": f"{roc_auc:.4f}",
            }
        )

        trained_models[name] = model

        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 70)
    print(results_df.to_string(index=False))

    return trained_models, results_df


def detailed_evaluation(model, X_test, y_test, feature_names):
    """Detailed model evaluation"""
    print("\n" + "=" * 70)
    print("DETAILED MODEL EVALUATION")
    print("=" * 70)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low Risk", "High Risk"],
        yticklabels=["Low Risk", "High Risk"],
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()
    plt.savefig("visualizations/04_confusion_matrix.png", dpi=300)
    print("✓ Saved: 04_confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(10, 8))
    plt.plot(
        fpr, tpr, color="darkorange", lw=3, label=f"ROC Curve (AUC = {roc_auc:.4f})"
    )
    plt.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier"
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve", fontsize=16, fontweight="bold")
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualizations/05_roc_curve.png", dpi=300)
    print("✓ Saved: 05_roc_curve.png")
    plt.close()

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": model.feature_importances_}
        ).sort_values("Importance", ascending=False)

        print("\nTop 15 Important Features:")
        print(importance_df.head(15).to_string(index=False))

        plt.figure(figsize=(12, 10))
        top_features = importance_df.head(20)
        sns.barplot(x="Importance", y="Feature", data=top_features, palette="viridis")
        plt.title("Top 20 Feature Importance", fontsize=16, fontweight="bold")
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()
        plt.savefig("visualizations/06_feature_importance.png", dpi=300)
        print("✓ Saved: 06_feature_importance.png")
        plt.close()


def save_model_files(model, scaler, feature_names):
    """Save model, scaler, and metadata"""
    print("\n" + "=" * 70)
    print("SAVING MODEL FILES")
    print("=" * 70)

    os.makedirs("models", exist_ok=True)

    with open("models/disease_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✓ Saved: models/disease_model.pkl")

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("✓ Saved: models/scaler.pkl")

    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    print("✓ Saved: models/feature_names.pkl")


def main():
    """Main training pipeline"""
    print("\n" + "=" * 70)
    print("DISEASE RISK PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 70)

    # Step 1: Load dataset
    df = load_dataset("data/raw/health_risk_dataset_full.csv")

    # Step 2: Explore dataset
    explore_dataset(df)

    # Step 3: Create visualizations
    create_visualizations(df)

    # Step 4: Prepare data
    print("\n" + "=" * 70)
    print("DATA PREPARATION")
    print("=" * 70)

    X = df.drop("disease_risk", axis=1)
    y = df["disease_risk"]
    feature_names = X.columns.tolist()

    print(f"\n✓ Total features: {len(feature_names)}")
    print(f"✓ Feature list: {', '.join(feature_names[:5])}... (showing first 5)")

    # Step 5: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n✓ Training set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")

    # Step 6: Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"\n✓ Features scaled using StandardScaler")

    # Step 7: Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    print(f"✓ Applied SMOTE - Balanced training set: {len(X_train_balanced)} samples")

    # Step 8: Train models
    trained_models, results_df = train_and_compare_models(
        X_train_balanced, X_test_scaled, y_train_balanced, y_test
    )

    # Step 9: Select best model (highest F1-score)
    best_idx = results_df["F1-Score"].astype(float).idxmax()
    best_model_name = results_df.loc[best_idx, "Model"]
    best_model = trained_models[best_model_name]

    print(f"\n{'=' * 70}")
    print(f"🏆 BEST MODEL: {best_model_name}")
    print(f"{'=' * 70}")

    # Step 10: Detailed evaluation
    detailed_evaluation(best_model, X_test_scaled, y_test, feature_names)

    # Step 11: Save model
    save_model_files(best_model, scaler, feature_names)

    # Final summary
    print("\n" + "=" * 70)
    print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nBest Model: {best_model_name}")
    print(f"F1-Score: {results_df.loc[best_idx, 'F1-Score']}")
    print(f"Accuracy: {results_df.loc[best_idx, 'Accuracy']}")
    print(f"\nModel files saved in: models/")
    print(f"Visualizations saved in: visualizations/")
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Review visualizations in visualizations/ folder")
    print("2. Run the Streamlit app: streamlit run app.py")
    print("3. Test predictions with data/sample/sample_input.csv")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
