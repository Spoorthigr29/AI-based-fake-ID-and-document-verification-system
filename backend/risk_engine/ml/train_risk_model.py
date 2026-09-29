"""
VerifyX AI — Risk Scoring Engine ML Training Pipeline.
Trains and compares Logistic Regression, Random Forest, and Gradient Boosting
classifiers on synthetic screening datasets for hackathon prototyping.
"""

import os
import sys
import json
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# -------------------------------------------------------------------------
# CONFIGURATION & REPRODUCIBILITY CONSTANTS
# -------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.20
CV_FOLDS = 5

NUMERICAL_FEATURES = [
    'document_quality',
    'ocr_confidence',
    'face_similarity',
    'tampering_probability',
    'identity_consistency'
]

BINARY_FEATURES = [
    'name_match',
    'dob_match',
    'id_format_valid',
    'address_match'
]

ALL_FEATURES = NUMERICAL_FEATURES + BINARY_FEATURES
TARGET_COLUMN = 'risk_category'
EXCLUDED_COLUMNS = ['verification_id', 'risk_score', 'data_type']

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DATASET_PATHS = [
    os.path.join(BACKEND_DIR, 'dataset', 'verifyx_120_records_training_dataset.csv'),
    os.path.join('backend', 'dataset', 'verifyx_120_records_training_dataset.csv'),
    os.path.join('dataset', 'verifyx_120_records_training_dataset.csv'),
    os.path.join(BACKEND_DIR, 'risk_engine', 'data', 'synthetic_risk_training_data.csv'),
    os.path.join(BACKEND_DIR, 'ml_models', 'data', 'synthetic_risk_training_data.csv'),
]

MODEL_OUTPUT_DIR = os.path.join(BACKEND_DIR, 'ml_models')
MODEL_FILE_PATH = os.path.join(MODEL_OUTPUT_DIR, 'risk_model.pkl')
METADATA_FILE_PATH = os.path.join(MODEL_OUTPUT_DIR, 'model_metadata.json')


def load_dataset() -> Tuple[pd.DataFrame, str]:
    """
    Step 1: Load synthetic dataset and inspect integrity.
    """
    selected_path = None
    for path in DATASET_PATHS:
        if os.path.exists(path):
            selected_path = path
            break

    if not selected_path:
        raise FileNotFoundError(f"Could not locate dataset in candidates: {DATASET_PATHS}")

    print("=" * 70)
    print("VERIFYX AI — RISK SCORING MODEL TRAINING PIPELINE")
    print("=" * 70)
    print(f"[*] Loading Dataset from: {selected_path}")

    df = pd.read_csv(selected_path)

    print("\n--- STEP 1: DATASET SUMMARY ---")
    print(f"Total Records (Rows): {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing Values:\n{df.isnull().sum().to_dict()}")
    print(f"Duplicate Rows: {df.duplicated().sum()}")
    print(f"\nClass Distribution ('{TARGET_COLUMN}'):")
    class_dist = df[TARGET_COLUMN].value_counts().to_dict()
    for cls_name, count in class_dist.items():
        print(f"  - {cls_name:15s}: {count:3d} ({count/len(df)*100:.1f}%)")

    return df, selected_path


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Step 2: Clean data, validate numerical ranges, and isolate features from target.
    """
    print("\n--- STEP 2: DATA PREPROCESSING ---")
    # Clean duplicates if any without modifying original CSV file
    df_clean = df.drop_duplicates().copy()

    # Fill any missing values in features with median / mode
    for num_col in NUMERICAL_FEATURES:
        if num_col in df_clean.columns and df_clean[num_col].isnull().any():
            df_clean[num_col] = df_clean[num_col].fillna(df_clean[num_col].median())

    for bin_col in BINARY_FEATURES:
        if bin_col in df_clean.columns:
            df_clean[bin_col] = df_clean[bin_col].astype(int)

    # Ensure no target leakage
    features_present = [f for f in ALL_FEATURES if f in df_clean.columns]
    print(f"Selected Model Features ({len(features_present)}): {features_present}")
    print(f"Excluded Leakage Columns: {EXCLUDED_COLUMNS}")

    X = df_clean[features_present]
    y = df_clean[TARGET_COLUMN]

    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build scikit-learn ColumnTransformer for feature scaling."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('bin', 'passthrough', BINARY_FEATURES)
        ]
    )
    return preprocessor


def train_and_evaluate_models(X_train, X_test, y_train, y_test) -> Tuple[Dict[str, Any], str]:
    """
    Step 3, 4, 5: Train multiple models, evaluate on test set & cross-validation,
    and select the optimal model based on macro F1 and class-wise recall.
    """
    print("\n--- STEP 3 & 4: MODEL TRAINING & EVALUATION ---")
    print(f"Training Set Size: {len(X_train)} samples")
    print(f"Testing Set Size:  {len(X_test)} samples")

    preprocessor = build_preprocessor()

    # Candidates
    candidates = {
        'Logistic Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight='balanced'
            ))
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=120,
                max_depth=6,
                random_state=RANDOM_STATE,
                class_weight='balanced'
            ))
        ]),
        'Gradient Boosting': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.08,
                max_depth=3,
                random_state=RANDOM_STATE
            ))
        ])
    }

    results = {}
    comparison_rows = []

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, pipeline in candidates.items():
        print(f"\nEvaluating: {name}...")

        # 5-fold Stratified Cross-Validation on training set
        cv_scores = cross_validate(
            pipeline, X_train, y_train,
            cv=skf,
            scoring=['accuracy', 'f1_macro', 'f1_weighted']
        )
        cv_acc = cv_scores['test_accuracy'].mean()
        cv_f1_macro = cv_scores['test_f1_macro'].mean()

        # Fit on full training set
        pipeline.fit(X_train, y_train)

        # Predict on holdout test set
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_mac = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weight = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_train))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_str = classification_report(y_test, y_pred, zero_division=0)

        results[name] = {
            'pipeline': pipeline,
            'cv_accuracy': float(cv_acc),
            'cv_f1_macro': float(cv_f1_macro),
            'test_accuracy': float(acc),
            'test_precision_macro': float(prec_macro),
            'test_recall_macro': float(rec_macro),
            'test_f1_macro': float(f1_mac),
            'test_f1_weighted': float(f1_weight),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'classification_report_str': report_str,
            'classes': list(np.unique(y_train))
        }

        comparison_rows.append({
            'Model': name,
            'CV Acc': f"{cv_acc*100:.1f}%",
            'CV Macro F1': f"{cv_f1_macro:.3f}",
            'Test Acc': f"{acc*100:.1f}%",
            'Macro Prec': f"{prec_macro:.3f}",
            'Macro Rec': f"{rec_macro:.3f}",
            'Macro F1': f"{f1_mac:.3f}",
            'Weighted F1': f"{f1_weight:.3f}"
        })

    # Print comparison table
    print("\n" + "=" * 80)
    print("--- STEP 5: MODEL COMPARISON & SELECTION TABLE ---")
    print("=" * 80)
    comp_df = pd.DataFrame(comparison_rows)
    print(comp_df.to_string(index=False))

    # Selection logic: prioritize Macro F1 and CV stability across imbalanced classes
    best_model_name = max(
        results.keys(),
        key=lambda k: (results[k]['test_f1_macro'] * 0.6) + (results[k]['cv_f1_macro'] * 0.4)
    )

    print("\n" + "=" * 80)
    print(f"[*] SELECTED MODEL: {best_model_name}")
    print(f"    Selected based on optimal Macro F1 ({results[best_model_name]['test_f1_macro']:.3f}) and balanced recall.")
    print("=" * 80)
    print(f"\nClassification Report for {best_model_name} (Test Set):")
    print(results[best_model_name]['classification_report_str'])

    return results, best_model_name


def save_model_and_metadata(results: Dict[str, Any], best_model_name: str, dataset_path: str, n_train: int, n_test: int):
    """
    Step 6: Persist selected scikit-learn Pipeline and metadata JSON.
    """
    print("\n--- STEP 6: SAVING MODEL ARTIFACTS ---")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    best_pipeline = results[best_model_name]['pipeline']
    best_metrics = {
        'test_accuracy': results[best_model_name]['test_accuracy'],
        'test_precision_macro': results[best_model_name]['test_precision_macro'],
        'test_recall_macro': results[best_model_name]['test_recall_macro'],
        'test_f1_macro': results[best_model_name]['test_f1_macro'],
        'test_f1_weighted': results[best_model_name]['test_f1_weighted'],
        'cv_accuracy': results[best_model_name]['cv_accuracy'],
        'cv_f1_macro': results[best_model_name]['cv_f1_macro'],
        'confusion_matrix': results[best_model_name]['confusion_matrix'],
        'classes': results[best_model_name]['classes']
    }

    # Save complete Pipeline
    joblib.dump(best_pipeline, MODEL_FILE_PATH)
    print(f"[OK] Saved Trained Pipeline: {MODEL_FILE_PATH}")

    # Build Metadata
    metadata = {
        'model_name': best_model_name,
        'training_date': datetime.now().isoformat(),
        'dataset_filename': os.path.basename(dataset_path),
        'dataset_path': dataset_path,
        'number_of_training_records': n_train,
        'number_of_testing_records': n_test,
        'feature_names': ALL_FEATURES,
        'numerical_features': NUMERICAL_FEATURES,
        'binary_features': BINARY_FEATURES,
        'target_name': TARGET_COLUMN,
        'classes': results[best_model_name]['classes'],
        'evaluation_metrics': best_metrics,
        'random_state': RANDOM_STATE,
        'models_compared': list(results.keys()),
        'disclaimer': (
            "The model achieved these performance metrics on the supplied synthetic evaluation dataset. "
            "These results represent performance on the synthetic dataset only and should not be "
            "interpreted as real-world fraud-detection performance."
        )
    }

    with open(METADATA_FILE_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Saved Model Metadata: {METADATA_FILE_PATH}")


def main():
    # 1. Load Data
    df, dataset_path = load_dataset()

    # 2. Preprocess Data
    X, y = preprocess_data(df)

    # 3. Train/Test Split (80/20 Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SPLIT_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # 4 & 5. Train, Compare & Select Model
    results, best_model_name = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # 6. Save Artifacts
    save_model_and_metadata(results, best_model_name, dataset_path, len(X_train), len(X_test))

    print("\n[OK] ML Risk Model Training and Persistence Completed Successfully.")


if __name__ == '__main__':
    main()
