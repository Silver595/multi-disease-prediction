# scripts/create_dataset.py
"""
Generate a realistic health risk prediction dataset with 5000 samples
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)


def generate_large_dataset(n_samples=5000):
    """Generate comprehensive health risk dataset"""

    print(f"Generating {n_samples} patient records...")

    # Demographics
    age = np.random.randint(18, 90, n_samples)
    gender = np.random.randint(0, 2, n_samples)  # 0=Female, 1=Male

    # Anthropometric measurements
    height_cm = np.where(
        gender == 1,
        np.random.normal(175, 7, n_samples),  # Male
        np.random.normal(162, 6, n_samples),
    )  # Female

    weight_kg = np.where(
        gender == 1,
        np.random.normal(80, 15, n_samples),  # Male
        np.random.normal(65, 12, n_samples),
    )  # Female

    bmi = weight_kg / ((height_cm / 100) ** 2)
    bmi = np.clip(bmi, 15, 50)

    # Vital signs (age and BMI dependent)
    blood_pressure_systolic = (
        110 + (age * 0.3) + (bmi * 0.9) + np.random.normal(0, 12, n_samples)
    )
    blood_pressure_systolic = np.clip(blood_pressure_systolic, 90, 220).astype(int)

    blood_pressure_diastolic = (
        70 + (age * 0.15) + (bmi * 0.5) + np.random.normal(0, 8, n_samples)
    )
    blood_pressure_diastolic = np.clip(blood_pressure_diastolic, 60, 130).astype(int)

    # Blood chemistry
    cholesterol_total = (
        160 + (age * 0.6) + (bmi * 1.5) + np.random.normal(0, 25, n_samples)
    )
    cholesterol_total = np.clip(cholesterol_total, 120, 400).astype(int)

    cholesterol_ldl = cholesterol_total * np.random.uniform(0.5, 0.7, n_samples)
    cholesterol_ldl = np.clip(cholesterol_ldl, 50, 300).astype(int)

    cholesterol_hdl = np.where(
        gender == 1,
        np.random.normal(45, 10, n_samples),  # Male
        np.random.normal(55, 12, n_samples),
    )  # Female
    cholesterol_hdl = np.clip(cholesterol_hdl, 20, 100).astype(int)

    triglycerides = (
        100 + (bmi - 22) * 6 + (age * 0.6) + np.random.normal(0, 40, n_samples)
    )
    triglycerides = np.clip(triglycerides, 40, 500).astype(int)

    # Glucose metabolism
    blood_sugar_fasting = (
        85 + (age * 0.25) + (bmi * 1.0) + np.random.normal(0, 18, n_samples)
    )
    blood_sugar_fasting = np.clip(blood_sugar_fasting, 65, 300).astype(int)

    hba1c = (
        5.0 + (blood_sugar_fasting - 90) * 0.025 + np.random.normal(0, 0.6, n_samples)
    )
    hba1c = np.clip(hba1c, 4.0, 14.0)

    # Cardiac
    heart_rate = np.random.normal(72, 11, n_samples)
    heart_rate = np.clip(heart_rate, 45, 130).astype(int)

    # Lifestyle factors
    smoking = np.random.binomial(1, 0.22, n_samples)  # 22% smokers

    # Exercise inversely related to BMI and age
    exercise_hours = np.maximum(
        0,
        6 - (bmi - 23) * 0.15 - (age - 40) * 0.03 + np.random.normal(0, 2.5, n_samples),
    )
    exercise_hours = np.clip(exercise_hours, 0, 20)

    # Physical activity categorization
    physical_activity = np.where(
        exercise_hours < 2,
        0,  # Sedentary
        np.where(
            exercise_hours < 5,
            1,  # Moderate
            2,
        ),
    )  # Active

    # Alcohol consumption
    alcohol_consumption = np.random.choice([0, 1, 2], n_samples, p=[0.35, 0.50, 0.15])

    # Sleep patterns
    sleep_hours = np.random.normal(7, 1.3, n_samples)
    sleep_hours = np.clip(sleep_hours, 3, 11)

    # Mental health
    stress_level = np.random.randint(1, 11, n_samples)

    # Dietary habits
    diet_quality = np.random.choice([0, 1, 2], n_samples, p=[0.25, 0.55, 0.20])

    # Medical history
    family_history = np.random.binomial(1, 0.35, n_samples)  # 35% family history
    previous_heart_condition = np.random.binomial(1, 0.12, n_samples)
    diabetes_history = np.random.binomial(1, 0.15, n_samples)
    chronic_kidney_disease = np.random.binomial(1, 0.08, n_samples)

    # Medications
    on_medication = np.random.binomial(1, 0.28, n_samples)

    # Kidney function
    creatinine = np.where(
        chronic_kidney_disease == 1,
        np.random.uniform(1.5, 3.5, n_samples),
        np.random.uniform(0.7, 1.3, n_samples),
    )

    # Liver function
    alt_enzyme = (
        20
        + (bmi - 22) * 0.8
        + alcohol_consumption * 8
        + np.random.normal(0, 10, n_samples)
    )
    alt_enzyme = np.clip(alt_enzyme, 10, 150).astype(int)

    # Calculate comprehensive risk score
    risk_score = (
        (age > 60) * 25
        + (age > 50) * 15
        + (age > 40) * 8
        + (gender == 1) * 6
        + (bmi > 35) * 30
        + (bmi > 30) * 20
        + (bmi > 25) * 10
        + (blood_pressure_systolic > 160) * 35
        + (blood_pressure_systolic > 140) * 25
        + (blood_pressure_systolic > 130) * 15
        + (blood_pressure_diastolic > 100) * 25
        + (blood_pressure_diastolic > 90) * 15
        + (cholesterol_total > 260) * 25
        + (cholesterol_total > 240) * 18
        + (cholesterol_total > 200) * 10
        + (cholesterol_ldl > 160) * 20
        + (cholesterol_ldl > 130) * 12
        + (cholesterol_hdl < 40) * 15
        + (triglycerides > 200) * 18
        + (triglycerides > 150) * 10
        + (blood_sugar_fasting > 126) * 30
        + (blood_sugar_fasting > 100) * 18
        + (hba1c > 6.5) * 28
        + (hba1c > 5.7) * 15
        + smoking * 20
        + (exercise_hours < 2) * 18
        + (exercise_hours < 4) * 10
        + family_history * 22
        + (physical_activity == 0) * 15
        + (alcohol_consumption == 2) * 12
        + (sleep_hours < 6) * 12
        + (stress_level > 8) * 10
        + (stress_level > 6) * 5
        + (diet_quality == 0) * 15
        + previous_heart_condition * 30
        + diabetes_history * 25
        + chronic_kidney_disease * 22
        + (heart_rate > 100) * 12
        + (heart_rate < 60) * 8
        + on_medication * 10
        + (creatinine > 1.5) * 18
        + (alt_enzyme > 40) * 10
        + np.random.normal(0, 8, n_samples)
    )

    # Binary classification (threshold calibrated for ~35% high risk)
    disease_risk = (risk_score > 75).astype(int)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "height_cm": np.round(height_cm, 1),
            "weight_kg": np.round(weight_kg, 1),
            "bmi": np.round(bmi, 1),
            "blood_pressure_systolic": blood_pressure_systolic,
            "blood_pressure_diastolic": blood_pressure_diastolic,
            "cholesterol_total": cholesterol_total,
            "cholesterol_ldl": cholesterol_ldl,
            "cholesterol_hdl": cholesterol_hdl,
            "triglycerides": triglycerides,
            "blood_sugar_fasting": blood_sugar_fasting,
            "hba1c": np.round(hba1c, 1),
            "heart_rate": heart_rate,
            "smoking": smoking,
            "exercise_hours": np.round(exercise_hours, 1),
            "physical_activity": physical_activity,
            "alcohol_consumption": alcohol_consumption,
            "sleep_hours": np.round(sleep_hours, 1),
            "stress_level": stress_level,
            "diet_quality": diet_quality,
            "family_history": family_history,
            "previous_heart_condition": previous_heart_condition,
            "diabetes_history": diabetes_history,
            "chronic_kidney_disease": chronic_kidney_disease,
            "on_medication": on_medication,
            "creatinine": np.round(creatinine, 2),
            "alt_enzyme": alt_enzyme,
            "disease_risk": disease_risk,
        }
    )

    return df


if __name__ == "__main__":
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/sample", exist_ok=True)

    # Generate full dataset
    print("=" * 70)
    print("HEALTH RISK DATASET GENERATOR")
    print("=" * 70)

    df = generate_large_dataset(5000)

    # Save full dataset
    df.to_csv("data/raw/health_risk_dataset_full.csv", index=False)
    print(f"\n✅ Full dataset saved: data/raw/health_risk_dataset_full.csv")

    # Create training dataset (80%)
    df_train = df.sample(frac=0.8, random_state=42)
    df_train.to_csv("data/processed/train_data.csv", index=False)
    print(
        f"✅ Training dataset saved: data/processed/train_data.csv ({len(df_train)} records)"
    )

    # Create test dataset (20%)
    df_test = df.drop(df_train.index)
    df_test.to_csv("data/processed/test_data.csv", index=False)
    print(
        f"✅ Test dataset saved: data/processed/test_data.csv ({len(df_test)} records)"
    )

    # Create sample for testing (100 records)
    df_sample = df.sample(n=100, random_state=42)
    df_sample_input = df_sample.drop("disease_risk", axis=1)
    df_sample_input.to_csv("data/sample/sample_input.csv", index=False)
    print(f"✅ Sample input saved: data/sample/sample_input.csv (100 records)")

    # Statistics
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"\nTotal Records: {len(df)}")
    print(f"Features: {len(df.columns) - 1}")
    print(f"\nFeature List:")
    for i, col in enumerate(df.columns[:-1], 1):
        print(f"  {i:2d}. {col}")

    print(f"\n{'=' * 70}")
    print("TARGET DISTRIBUTION")
    print(f"{'=' * 70}")
    print(df["disease_risk"].value_counts())
    print(f"\nPercentage:")
    dist = df["disease_risk"].value_counts(normalize=True) * 100
    print(f"  Low Risk (0):  {dist[0]:.2f}%")
    print(f"  High Risk (1): {dist[1]:.2f}%")

    print(f"\n{'=' * 70}")
    print("SAMPLE DATA (First 5 rows)")
    print(f"{'=' * 70}")
    print(df.head().to_string())

    print(f"\n{'=' * 70}")
    print("✅ DATASET GENERATION COMPLETED!")
    print(f"{'=' * 70}\n")
