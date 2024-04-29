"""Utility functions for calculating additional metrics and plotting confusion matrices with seaborn instead of wandb."""
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%

simple_acc = pd.read_csv('../../results/simple_acc.csv')
accuracies = simple_acc['Simple permutation (window: 7, stride:11) - Validation Accuracy']
accuracies.mean()

# %%
trans_only = pd.read_csv('../../results/transformer_only_acc.csv')
transformer_only_accuracies = trans_only['transformer only - Validation Accuracy']
transformer_only_accuracies.mean()

# %%
TP, FP, FN, TN = 100, 10, 5, 200

confusion_matrix = np.array([[TP, FP],
                             [FN, TN]])


class_labels = ['Left hand', 'Right hand']


plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels, cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# %%
