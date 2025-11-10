"""
Credit Card Fraud Detection - Logistic Regression (OPTIMIZED)
Faster version with better hyperparameter choices
Paper: Parameter C tuned using RandomizedSearchCV
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score)
import pickle
import time
import warnings
import os
warnings.filterwarnings('ignore')

# Create folders
os.makedirs('plots_ml', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

# Set random seed
np.random.seed(42)

print("="*80)
print("CREDIT CARD FRAUD DETECTION - LOGISTIC REGRESSION (OPTIMIZED)")
print("Paper Parameters: Parameter C tuned using RandomizedSearchCV")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATASET
# ============================================================================
print("\n[1] Loading Dataset...")
df = pd.read_csv('creditcard.csv')
print(f"Dataset Shape: {df.shape}")
print(f"Fraudulent Transactions: {df['Class'].sum()} ({df['Class'].sum()/len(df)*100:.3f}%)")

# ============================================================================
# STEP 2: DATA PREPARATION WITH STANDARDSCALER
# ============================================================================
print("\n[2] Data Preparation...")
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training Set: {X_train.shape}")
print(f"Testing Set: {X_test.shape}")

# Apply StandardScaler
print("\nApplying StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling completed")

# ============================================================================
# STEP 3: OPTIMIZED RANDOMIZEDSEARCHCV (FASTER)
# ============================================================================
print("\n[3] Hyperparameter Tuning with RandomizedSearchCV (Optimized)...")
print("Using only fast solvers: lbfgs and liblinear")

# Optimized parameter distribution (removed slow 'saga' solver)
param_dist = {
    'C': [0.01, 0.1, 1, 10, 100],  # Reduced options
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']  # Only fast solvers
}

# Initialize base model with increased max_iter
lr_base = LogisticRegression(max_iter=1000, random_state=42)

# Faster RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=lr_base,
    param_distributions=param_dist,
    n_iter=8,  # Reduced from 10
    cv=3,
    scoring='f1',
    random_state=42,
    n_jobs=-1,
    verbose=2
)

print("Running RandomizedSearchCV (should take 2-5 minutes)...")
search_start = time.time()
random_search.fit(X_train_scaled, y_train)
search_time = time.time() - search_start

print(f"\nRandomizedSearchCV completed in {search_time:.2f} seconds ({search_time/60:.1f} minutes)")
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best CV F1-Score: {random_search.best_score_:.4f}")

lr_model = random_search.best_estimator_

# ============================================================================
# STEP 4: TRAIN FINAL MODEL
# ============================================================================
print("\n[4] Training Final Model with Best Parameters...")
train_start = time.time()
lr_model.fit(X_train_scaled, y_train)
train_time = time.time() - train_start
print(f"Training completed in {train_time:.2f} seconds")

# ============================================================================
# STEP 5: MODEL PREDICTION
# ============================================================================
print("\n[5] Making Predictions...")
test_start = time.time()
y_pred = lr_model.predict(X_test_scaled)
y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
test_time = time.time() - test_start
print(f"Prediction completed in {test_time:.4f} seconds")

# ============================================================================
# STEP 6: EVALUATION METRICS
# ============================================================================
print("\n[6] Evaluation Metrics:")
print("="*80)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1-Score:  {f1*100:.2f}%")
print(f"ROC-AUC:   {roc_auc*100:.2f}%")

print("\n" + "="*80)
print("Classification Report:")
print("="*80)
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

# ============================================================================
# STEP 7: CONFUSION MATRIX
# ============================================================================
print("\n[7] Generating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'])
plt.title('Logistic Regression - Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('plots_ml/logistic_regression_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nSaved: plots_ml/logistic_regression_confusion_matrix.png")
plt.close()

# ============================================================================
# STEP 8: ROC CURVE
# ============================================================================
print("\n[8] Generating ROC Curve...")
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='orange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Logistic Regression - ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots_ml/logistic_regression_roc_curve.png', dpi=300, bbox_inches='tight')
print("Saved: plots_ml/logistic_regression_roc_curve.png")
plt.close()

# ============================================================================
# STEP 9: PRECISION-RECALL CURVE
# ============================================================================
print("\n[9] Generating Precision-Recall Curve...")
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, color='darkorange', lw=2, 
         label=f'PR curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Logistic Regression - Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots_ml/logistic_regression_precision_recall_curve.png', dpi=300, bbox_inches='tight')
print("Saved: plots_ml/logistic_regression_precision_recall_curve.png")
plt.close()

# ============================================================================
# STEP 10: FEATURE COEFFICIENTS
# ============================================================================
print("\n[10] Analyzing Feature Coefficients...")
feature_coef = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_[0],
    'Abs_Coefficient': np.abs(lr_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\nTop 15 Important Features:")
print(feature_coef.head(15)[['Feature', 'Coefficient', 'Abs_Coefficient']].to_string(index=False))

top_15 = feature_coef.head(15)
plt.figure(figsize=(10, 6))
colors = ['red' if x < 0 else 'green' for x in top_15['Coefficient']]
plt.barh(top_15['Feature'][::-1], top_15['Coefficient'][::-1], color=colors[::-1])
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Logistic Regression - Top 15 Feature Coefficients', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig('plots_ml/logistic_regression_feature_coefficients.png', dpi=300, bbox_inches='tight')
print("\nSaved: plots_ml/logistic_regression_feature_coefficients.png")
plt.close()

# ============================================================================
# STEP 11: SAVE MODEL AND SCALER
# ============================================================================
print("\n[11] Saving Model and Scaler...")

with open('saved_models/logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("✓ Model saved as 'saved_models/logistic_regression_model.pkl'")

with open('saved_models/logistic_regression_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved as 'saved_models/logistic_regression_scaler.pkl'")

# ============================================================================
# STEP 12: SAVE RESULTS
# ============================================================================
print("\n[12] Saving Results...")

results_df = pd.DataFrame({
    'Model': ['Logistic Regression'],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1-Score': [f1],
    'ROC-AUC': [roc_auc],
    'Train_Time': [train_time],
    'Test_Time': [test_time],
    'Best_C': [random_search.best_params_['C']],
    'Best_Solver': [random_search.best_params_['solver']],
    'CV_Time': [search_time]
})

if os.path.exists('results_ml.csv'):
    existing_df = pd.read_csv('results_ml.csv')
    for col in results_df.columns:
        if col not in existing_df.columns:
            existing_df[col] = np.nan
    for col in existing_df.columns:
        if col not in results_df.columns:
            results_df[col] = np.nan
    results_df = pd.concat([existing_df, results_df], ignore_index=True)

results_df.to_csv('results_ml.csv', index=False)
print("Saved: results_ml.csv")

print("\n" + "="*80)
print("MODEL SUMMARY")
print("="*80)
print(f"Best Parameters: C={random_search.best_params_['C']}, solver={random_search.best_params_['solver']}")
print(f"\nPerformance Metrics:")
print(f"  - Accuracy:  {accuracy*100:.2f}%")
print(f"  - Precision: {precision*100:.2f}%")
print(f"  - Recall:    {recall*100:.2f}%")
print(f"  - F1-Score:  {f1*100:.2f}%")
print(f"  - ROC-AUC:   {roc_auc*100:.2f}%")
print(f"\nPaper Benchmark: 99.91% accuracy, 73.56% F1-score")
print(f"\nTiming: Search={search_time:.1f}s, Train={train_time:.2f}s, Test={test_time:.4f}s")

print("\n" + "="*80)
print("LOGISTIC REGRESSION COMPLETED SUCCESSFULLY")
print("="*80)
