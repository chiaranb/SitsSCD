import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ---------------- Helper: plot confusion matrix as image ----------------
def plot_confusion_matrix_image(y_true=None, y_pred=None, class_names=None, title="Confusion Matrix", matrix=None):
    if matrix is None:
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)), normalize='true')
    else:
        cm = matrix

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    return fig

def compute_iou_confusion_matrix(y_true_list, y_pred_list, num_classes):
    conf_mat = np.zeros((num_classes, num_classes), dtype=float)
    for t, p in zip(y_true_list, y_pred_list):
        conf_mat[t, p] += 1

    tp = np.diag(conf_mat)
    fp = conf_mat.sum(axis=0) - tp
    fn = conf_mat.sum(axis=1) - tp
    denom = tp + fp + fn + 1e-8

    iou_mat = np.zeros_like(conf_mat)
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                iou_mat[i, i] = tp[i] / denom[i]
            else:
                iou_mat[i, j] = (conf_mat[i, j] + conf_mat[j, i]) / denom[i]

    return iou_mat 