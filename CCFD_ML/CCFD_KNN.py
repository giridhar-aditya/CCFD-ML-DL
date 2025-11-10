"""
Credit Card Fraud Detection - K-Nearest Neighbors with Model Saving
Strictly following paper parameters: k=5 (n_neighbors=5)
Paper Result: 99.95% Accuracy, 85.71% F1-Score
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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
print("CREDIT CARD FRAUD DETECTION - K-NEAREST NEIGHBORS (KNN)")
print("Paper Parameters: k=5 (n_neighbors=5)")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATASET
# ============================================================================
print("\n[1] Loading Dataset...")
df = pd.read_csv('creditcard.csv')
print(f"Dataset Shape: {df.shape}")
print(f"Fraudulent Transactions: {df['Class'].sum()} ({df['Class'].sum()/len(df)*100:.3f}%)")
print(f"Legitimate Transactions: {len(df) - df['Class'].sum()}")

# ============================================================================
# STEP 2: DATA PREPARATION WITH STANDARDSCALER
# ============================================================================
print("\n[2] Data Preparation...")
print("NOTE: KNN is a distance-based algorithm and requires feature scaling")

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training Set: {X_train.shape}")
print(f"Testing Set: {X_test.shape}")
print(f"Training Fraud Cases: {y_train.sum()}")
print(f"Testing Fraud Cases: {y_test.sum()}")

# Apply StandardScaler
print("\nApplying StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling completed")

# ============================================================================
# STEP 3: BUILD AND TRAIN MODEL
# ============================================================================
print("\n[3] Building KNN Model...")
print("Parameters: n_neighbors=5")

# Initialize model with paper parameters
knn_model = KNeighborsClassifier(
    n_neighbors=5,
    n_jobs=-1
)

# Train model
print("Training model...")
train_start = time.time()
knn_model.fit(X_train_scaled, y_train)
train_time = time.time() - train_start
print(f"Training completed in {train_time:.2f} seconds")

# ============================================================================
# STEP 4: MODEL PREDICTION
# ============================================================================
print("\n[4] Making Predictions...")
test_start = time.time()
y_pred = knn_model.predict(X_test_scaled)
y_pred_proba = knn_model.predict_proba(X_test_scaled)[:, 1]
test_time = time.time() - test_start
print(f"Prediction completed in {test_time:.4f} seconds")

# ============================================================================
# STEP 5: EVALUATION METRICS
# ============================================================================
print("\n[5] Evaluation Metrics:")
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
# STEP 6: CONFUSION MATRIX
# ============================================================================
print("\n[6] Generating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print(f"\nTrue Negatives:  {cm[0, 0]}")
print(f"False Positives: {cm[0, 1]}")
print(f"False Negatives: {cm[1, 0]}")
print(f"True Positives:  {cm[1, 1]}")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'])
plt.title('KNN (k=5) - Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('plots_ml/knn_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nSaved: plots_ml/knn_confusion_matrix.png")
plt.close()

# ============================================================================
# STEP 7: ROC CURVE
# ============================================================================
print("\n[7] Generating ROC Curve...")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('KNN (k=5) - ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots_ml/knn_roc_curve.png', dpi=300, bbox_inches='tight')
print("Saved: plots_ml/knn_roc_curve.png")
plt.close()

# ============================================================================
# STEP 8: PRECISION-RECALL CURVE
# ============================================================================
print("\n[8] Generating Precision-Recall Curve...")

precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, color='darkgreen', lw=2, 
         label=f'PR curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('KNN (k=5) - Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots_ml/knn_precision_recall_curve.png', dpi=300, bbox_inches='tight')
print("Saved: plots_ml/knn_precision_recall_curve.png")
plt.close()

# ============================================================================
# STEP 9: SAVE MODEL AND SCALER
# ============================================================================
print("\n[9] Saving Model and Scaler...")

# Save KNN model
with open('saved_models/knn_model.pkl', 'wb') as f:
    pickle.dump(knn_model, f)
print("✓ Model saved as 'saved_models/knn_model.pkl'")

# Save scaler
with open('saved_models/knn_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved as 'saved_models/knn_scaler.pkl'")

# ============================================================================
# STEP 10: SAVE RESULTS
# ============================================================================
print("\n[10] Saving Results...")

results_df = pd.DataFrame({
    'Model': ['KNN'],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1-Score': [f1],
    'ROC-AUC': [roc_auc],
    'Train_Time': [train_time],
    'Test_Time': [test_time],
    'Parameters': ['n_neighbors=5']
})

# Append to existing results
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
print("KNN MODEL COMPLETED SUCCESSFULLY")
print("="*80)
print("\nGenerated Files:")
print("  1. plots_ml/knn_confusion_matrix.png")
print("  2. plots_ml/knn_roc_curve.png")
print("  3. plots_ml/knn_precision_recall_curve.png")
print("  4. saved_models/knn_model.pkl")
print("  5. saved_models/knn_scaler.pkl")
print("  6. results_ml.csv (updated)")
print("="*80)
