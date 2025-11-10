"""
Credit Card Fraud Detection - XGBoost with Model Saving
Strictly following paper parameters: Default XGBoost
Paper Result: 99.94% Accuracy, 84.49% F1-Score (Best ML Model)
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
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
print("CREDIT CARD FRAUD DETECTION - XGBOOST")
print("Paper Parameters: Default XGBoost")
print("Paper Result: 99.94% Accuracy, 84.49% F1 (Best ML Model)")
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
# STEP 2: DATA PREPARATION (NO STANDARDSCALER FOR TREE-BASED)
# ============================================================================
print("\n[2] Data Preparation...")
print("NOTE: No StandardScaler used (tree-based algorithm)")


# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


print(f"Training Set: {X_train.shape}")
print(f"Testing Set: {X_test.shape}")
print(f"Training Fraud Cases: {y_train.sum()}")
print(f"Testing Fraud Cases: {y_test.sum()}")


# Calculate scale_pos_weight for class imbalance
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(f"Scale Pos Weight: {scale_pos_weight:.2f}")


# ============================================================================
# STEP 3: BUILD AND TRAIN MODEL
# ============================================================================
print("\n[3] Building XGBoost Model...")
print(f"Parameters: scale_pos_weight={scale_pos_weight:.2f}, random_state=42")


# Initialize model
xgb_model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)


# Train model
print("Training model...")
train_start = time.time()
xgb_model.fit(X_train, y_train)
train_time = time.time() - train_start
print(f"Training completed in {train_time:.2f} seconds")


# ============================================================================
# STEP 4: MODEL PREDICTION
# ============================================================================
print("\n[4] Making Predictions...")
test_start = time.time()
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
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
avg_precision = average_precision_score(y_test, y_pred_proba)


print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1-Score:  {f1*100:.2f}%")
print(f"ROC-AUC:   {roc_auc*100:.2f}%")
print(f"Average Precision: {avg_precision*100:.2f}%")


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
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=False,
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'])
plt.title('XGBoost - Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('plots_ml/xgboost_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nSaved: plots_ml/xgboost_confusion_matrix.png")
plt.close()


# ============================================================================
# STEP 7: ROC CURVE
# ============================================================================
print("\n[7] Generating ROC Curve...")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='teal', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('XGBoost - ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots_ml/xgboost_roc_curve.png', dpi=300, bbox_inches='tight')
print("Saved: plots_ml/xgboost_roc_curve.png")
plt.close()


# ============================================================================
# STEP 8: PRECISION-RECALL CURVE (NEW)
# ============================================================================
print("\n[8] Generating Precision-Recall Curve...")
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)


plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, color='darkblue', lw=2, 
         label=f'PR curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('XGBoost - Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots_ml/xgboost_precision_recall_curve.png', dpi=300, bbox_inches='tight')
print("Saved: plots_ml/xgboost_precision_recall_curve.png")
plt.close()


# ============================================================================
# STEP 9: FEATURE IMPORTANCE
# ============================================================================
print("\n[9] Analyzing Feature Importance...")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False).head(15)


print("\nTop 15 Important Features:")
print(feature_importance.to_string(index=False))


plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'][::-1], feature_importance['Importance'][::-1])
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('XGBoost - Top 15 Feature Importances', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots_ml/xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
print("\nSaved: plots_ml/xgboost_feature_importance.png")
plt.close()


# ============================================================================
# STEP 10: SAVE MODEL
# ============================================================================
print("\n[10] Saving Model...")


# Save XGBoost model
with open('saved_models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("✓ Model saved as 'saved_models/xgboost_model.pkl'")


# Also save in XGBoost native format
xgb_model.save_model('saved_models/xgboost_model.json')
print("✓ Model saved as 'saved_models/xgboost_model.json' (XGBoost native format)")


# ============================================================================
# STEP 11: SAVE RESULTS
# ============================================================================
print("\n[11] Saving Results...")


results_df = pd.DataFrame({
    'Model': ['XGBoost'],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1-Score': [f1],
    'ROC-AUC': [roc_auc],
    'Avg_Precision': [avg_precision],
    'Train_Time': [train_time],
    'Test_Time': [test_time],
    'Parameters': [f'scale_pos_weight={scale_pos_weight:.2f}']
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
print("XGBOOST MODEL COMPLETED SUCCESSFULLY")
print("="*80)
print("\nGenerated Files:")
print("  1. plots_ml/xgboost_confusion_matrix.png")
print("  2. plots_ml/xgboost_roc_curve.png")
print("  3. plots_ml/xgboost_precision_recall_curve.png")
print("  4. plots_ml/xgboost_feature_importance.png")
print("  5. saved_models/xgboost_model.pkl")
print("  6. saved_models/xgboost_model.json")
print("  7. results_ml.csv (updated)")
print("="*80)
