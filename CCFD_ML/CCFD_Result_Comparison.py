"""
Complete ML Models Comparison - Recreating Paper Figures
Generates Figure 6 (Confusion Matrices), Figure 7 (Case Statistics), Figure 8 (Comparative Analysis)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.gridspec import GridSpec

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

# Create output folder
os.makedirs('plots_ml', exist_ok=True)

print("="*80)
print("ML MODELS COMPARISON - RECREATING PAPER FIGURES")
print("="*80)

# ============================================================================
# LOAD DATA AND RESULTS
# ============================================================================
print("\n[1] Loading Data and Results...")

# Load dataset for statistics
df = pd.read_csv('creditcard.csv')
print(f"Dataset loaded: {df.shape}")

# Load ML results
results_df = pd.read_csv('results_ml.csv')
print(f"ML Results loaded: {len(results_df)} models")
print("\nModels in results:")
print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))

# ============================================================================
# FIGURE 6: CONFUSION MATRICES OF ML ALGORITHMS (6 subplots in 3x2 grid)
# ============================================================================
print("\n[2] Generating Figure 6: Confusion Matrices of ML Algorithms...")

# Note: We'll create a composite from individual confusion matrix images
# Since we already have individual confusion matrices, we'll create a grid layout

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

models_order = ['Decision Tree', 'KNN', 'Logistic Regression', 'SVM', 'XGBoost', 'Random Forest']
confusion_files = [
    'decision_tree_confusion_matrix.png',
    'knn_confusion_matrix.png',
    'logistic_regression_confusion_matrix.png',
    'svm_confusion_matrix.png',
    'xgboost_confusion_matrix.png',
    'random_forest_confusion_matrix.png'
]

for idx, (model, cm_file) in enumerate(zip(models_order, confusion_files)):
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    
    # Try to load and display the confusion matrix image
    cm_path = f'plots_ml/{cm_file}'
    if os.path.exists(cm_path):
        img = plt.imread(cm_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'{model}', fontsize=12, fontweight='bold', pad=10)
    else:
        ax.text(0.5, 0.5, f'{model}\nConfusion Matrix\nNot Found', 
                ha='center', va='center', fontsize=10)
        ax.axis('off')

plt.suptitle('Figure 6: Confusion Metrics of Machine Learning Algorithms', 
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig('plots_ml/figure_6_confusion_matrices_grid.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots_ml/figure_6_confusion_matrices_grid.png")
plt.close()

# ============================================================================
# FIGURE 7: CASE COUNT STATISTICS FOR FRAUD AND NON-FRAUD TRANSACTIONS
# ============================================================================
print("\n[3] Generating Figure 7: Case Count Statistics...")

# Calculate statistics
fraud_df = df[df['Class'] == 1]
legit_df = df[df['Class'] == 0]

fraud_stats = fraud_df['Amount'].describe()
legit_stats = legit_df['Amount'].describe()

statistics = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
fraud_values = [fraud_stats[stat] for stat in statistics]
legit_values = [legit_stats[stat] for stat in statistics]

# Create line plot
fig, ax = plt.subplots(figsize=(10, 6))

x_positions = np.arange(len(statistics))
ax.plot(x_positions, legit_values, marker='o', color='green', linewidth=2, 
        markersize=8, label='NON-FRAUD CASE AMOUNT STATS')
ax.plot(x_positions, fraud_values, marker='o', color='blue', linewidth=2, 
        markersize=8, label='FRAUD CASE AMOUNT STATS')

ax.set_xlabel('Statistics', fontsize=12, fontweight='bold')
ax.set_ylabel('Amount', fontsize=12, fontweight='bold')
ax.set_title('CASE AMOUNT STATISTICS', fontsize=14, fontweight='bold')
ax.set_xticks(x_positions)
ax.set_xticklabels(['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots_ml/figure_7_case_statistics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots_ml/figure_7_case_statistics.png")
plt.close()

# ============================================================================
# TABLE 5: THE ACCURACY AND F1-SCORE OF MACHINE LEARNING ALGORITHMS
# ============================================================================
print("\n[4] Generating Table 5: Accuracy and F1-Score Table...")

# Reorder to match paper
models_order = ['Decision Tree', 'KNN', 'Logistic Regression', 'SVM', 'Random Forest', 'XGBoost']
table_data = []

for model in models_order:
    if model in results_df['Model'].values:
        row = results_df[results_df['Model'] == model].iloc[0]
        table_data.append([
            model,
            f"{row['Accuracy']*100:.2f}",
            f"{row['F1-Score']*100:.2f}"
        ])

# Create table figure
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=table_data,
                colLabels=['Algorithm Name', 'Accuracy (%)', 'F1-Score (%)'],
                cellLoc='center',
                loc='center',
                colWidths=[0.5, 0.25, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Style header
for i in range(3):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data) + 1):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')

plt.title('TABLE 5. The accuracy and F1-score of machine learning algorithms', 
          fontsize=14, fontweight='bold', pad=20)
plt.savefig('plots_ml/table_5_ml_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots_ml/table_5_ml_results.png")
plt.close()

# ============================================================================
# FIGURE 8: COMPARATIVE ANALYSIS OF MACHINE LEARNING ALGORITHMS
# ============================================================================
print("\n[5] Generating Figure 8: Comparative Analysis (Bar Chart)...")

# Prepare data (use abbreviations to match paper)
model_abbr = {
    'Decision Tree': 'DT',
    'KNN': 'KNN',
    'Logistic Regression': 'LR',
    'SVM': 'SVM',
    'Random Forest': 'RFT',
    'XGBoost': 'XGB'
}

models = []
accuracy_scores = []
f1_scores = []

for model in models_order:
    if model in results_df['Model'].values:
        row = results_df[results_df['Model'] == model].iloc[0]
        models.append(model_abbr.get(model, model))
        accuracy_scores.append(row['Accuracy'])
        f1_scores.append(row['F1-Score'])

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))

bars1 = ax.bar(x - width/2, accuracy_scores, width, label='Accuracy Score', 
               color='steelblue', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', 
               color='coral', edgecolor='black', linewidth=1.2)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Models', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy and F1 Score', fontsize=12, fontweight='bold')
ax.set_title('ACCURACY SCORE OF MACHINE LEARNING MODELS', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('plots_ml/figure_8_comparative_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots_ml/figure_8_comparative_analysis.png")
plt.close()

# ============================================================================
# ADDITIONAL: ALL METRICS COMPARISON (GROUPED BAR CHART)
# ============================================================================
print("\n[6] Generating Additional Comprehensive Metrics Comparison...")

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(models))
width = 0.15

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, metric in enumerate(metrics):
    values = []
    for model in models_order:
        if model in results_df['Model'].values:
            row = results_df[results_df['Model'] == model].iloc[0]
            values.append(row[metric])
    
    offset = width * (i - 2)
    ax.bar(x + offset, values, width, label=metric, color=colors[i])

ax.set_xlabel('Models', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Comprehensive ML Models Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([model_abbr.get(m, m) for m in models_order], fontsize=11)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plots_ml/comprehensive_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots_ml/comprehensive_metrics_comparison.png")
plt.close()

# ============================================================================
# HEATMAP: PERFORMANCE MATRIX
# ============================================================================
print("\n[7] Generating Performance Heatmap...")

heatmap_models = [model_abbr.get(m, m) for m in models_order]
heatmap_data = []

for model in models_order:
    if model in results_df['Model'].values:
        row = results_df[results_df['Model'] == model].iloc[0]
        heatmap_data.append([
            row['Accuracy'],
            row['Precision'],
            row['Recall'],
            row['F1-Score'],
            row['ROC-AUC']
        ])

heatmap_df = pd.DataFrame(heatmap_data, 
                          columns=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                          index=heatmap_models)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_df, annot=True, fmt='.4f', cmap='RdYlGn', 
            cbar_kws={'label': 'Score'}, vmin=0, vmax=1, linewidths=0.5,
            linecolor='black', ax=ax)
plt.title('ML Models Performance Heatmap', fontsize=14, fontweight='bold')
plt.xlabel('Metrics', fontsize=12, fontweight='bold')
plt.ylabel('Models', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('plots_ml/performance_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots_ml/performance_heatmap.png")
plt.close()

# ============================================================================
# TRAINING TIME COMPARISON
# ============================================================================
print("\n[8] Generating Training Time Comparison...")

train_times = []
for model in models_order:
    if model in results_df['Model'].values:
        row = results_df[results_df['Model'] == model].iloc[0]
        train_times.append(row['Train_Time'])

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(heatmap_models, train_times, color='mediumseagreen', edgecolor='black')

# Add value labels
for i, (bar, time) in enumerate(zip(bars, train_times)):
    ax.text(time + max(train_times)*0.02, i, f'{time:.2f}s', 
            va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Model Training Time Comparison', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('plots_ml/training_time_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots_ml/training_time_comparison.png")
plt.close()

# ============================================================================
# SUMMARY STATISTICS TABLE
# ============================================================================
print("\n[9] Generating Summary Statistics...")

summary_stats = pd.DataFrame({
    'Model': heatmap_models,
    'Accuracy (%)': [f"{a*100:.2f}" for a in [row['Accuracy'] for _, row in results_df.iterrows()]],
    'Precision (%)': [f"{p*100:.2f}" for p in [row['Precision'] for _, row in results_df.iterrows()]],
    'Recall (%)': [f"{r*100:.2f}" for r in [row['Recall'] for _, row in results_df.iterrows()]],
    'F1-Score (%)': [f"{f*100:.2f}" for f in [row['F1-Score'] for _, row in results_df.iterrows()]],
    'ROC-AUC (%)': [f"{a*100:.2f}" for a in [row['ROC-AUC'] for _, row in results_df.iterrows()]]
})

print("\nSummary Statistics:")
print(summary_stats.to_string(index=False))

# Save to CSV
summary_stats.to_csv('ml_models_summary_statistics.csv', index=False)
print("\n✓ Saved: ml_models_summary_statistics.csv")

# ============================================================================
# FIND BEST MODELS
# ============================================================================
print("\n[10] Best Model Analysis:")
print("="*80)

best_accuracy_idx = results_df['Accuracy'].idxmax()
best_f1_idx = results_df['F1-Score'].idxmax()
best_precision_idx = results_df['Precision'].idxmax()
best_recall_idx = results_df['Recall'].idxmax()

print(f"Best Accuracy:  {results_df.loc[best_accuracy_idx, 'Model']} ({results_df.loc[best_accuracy_idx, 'Accuracy']*100:.2f}%)")
print(f"Best F1-Score:  {results_df.loc[best_f1_idx, 'Model']} ({results_df.loc[best_f1_idx, 'F1-Score']*100:.2f}%)")
print(f"Best Precision: {results_df.loc[best_precision_idx, 'Model']} ({results_df.loc[best_precision_idx, 'Precision']*100:.2f}%)")
print(f"Best Recall:    {results_df.loc[best_recall_idx, 'Model']} ({results_df.loc[best_recall_idx, 'Recall']*100:.2f}%)")

print("\n" + "="*80)
print("ALL PAPER FIGURES AND COMPARISONS GENERATED SUCCESSFULLY!")
print("="*80)
print("\nGenerated Files:")
print("  1. plots_ml/figure_6_confusion_matrices_grid.png")
print("  2. plots_ml/figure_7_case_statistics.png")
print("  3. plots_ml/table_5_ml_results.png")
print("  4. plots_ml/figure_8_comparative_analysis.png")
print("  5. plots_ml/comprehensive_metrics_comparison.png")
print("  6. plots_ml/performance_heatmap.png")
print("  7. plots_ml/training_time_comparison.png")
print("  8. ml_models_summary_statistics.csv")
print("="*80)
