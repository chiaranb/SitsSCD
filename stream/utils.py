import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import imageio
import os
from collections import Counter
import csv

# ---------------- Helper: plot confusion matrix as image ----------------
def plot_confusion_matrix_image(
    y_true=None,
    y_pred=None,
    class_names=None,
    title="Confusion Matrix",
    matrix=None
):
    if matrix is None:
        # Confusion matrix grezza (conteggi)
        cm_counts = confusion_matrix(
            y_true, y_pred, labels=range(len(class_names))
        )
        # Confusion matrix normalizzata per riga (true)
        cm_norm = confusion_matrix(
            y_true, y_pred, labels=range(len(class_names)), normalize="true"
        )

        supports = np.bincount(
            np.array(y_true), minlength=len(class_names)
        )
    else:
        cm_counts = matrix
        cm_norm = matrix
        supports = None

    # Costruisci le annotazioni: "0.83\n(124)"
    annot = np.empty_like(cm_norm, dtype=object)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            annot[i, j] = f"{cm_norm[i, j]:.3f}\n({cm_counts[i, j]})"

    if supports is not None:
        y_labels = [
            f"{name}\n({supports[i]})" for i, name in enumerate(class_names)
        ]
    else:
        y_labels = class_names

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=y_labels,
        ax=ax,
        cbar=True
    )

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

# ---------------- Helper: initialize CSV files ----------------
def init_csv_files(misclass_file, wrong_bc_file, wrong_sc_file):
    if not os.path.exists(misclass_file):
        with open(misclass_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_id", "timestamp",
                "sits_id", "patch_id",
                "y_true", "y_pred"
            ])

    if not os.path.exists(wrong_bc_file):
        with open(wrong_bc_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_id", "timestamp",
                "sits_id", "patch_id",
                "y_true_t-1", "y_true_t",
                "y_pred_t-1", "y_pred_t"
            ])

    if not os.path.exists(wrong_sc_file):
        with open(wrong_sc_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_id", "timestamp",
                "sits_id", "patch_id",
                "y_true_t-1", "y_true_t",
                "y_pred_t-1", "y_pred_t"
            ])

# ---------------- Timeline plot ----------------
def plot_patch_timeline(timestamps, y_true_seq, y_pred_seq, class_names, title):
    x = list(range(len(timestamps)))

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.plot(x, y_true_seq, "s-", label="GT")
    ax.plot(x, y_pred_seq, "o--", label="Pred")

    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xticks(x)
    ax.set_xticklabels(
        month_labels[:len(x)],
        fontsize=7
    )
    ax.grid(True, axis="x", which="major", linestyle="--", alpha=0.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    ax.set_xlabel("Month")
    ax.set_ylabel("Class")
    ax.set_title(title)
    ax.legend(frameon=False)

    plt.tight_layout()
    return fig

def compute_entropy(probs):
    """Calcola entropia normalizzata (0=certo, 1=massima incertezza)"""
    return -np.sum(probs * np.log(probs + 1e-10)) / np.log(len(probs))

def save_patch_image(
    patch_tensor,        # [C, H, W]
    sits_id,
    patch_id,
    timestamp,
    out_dir,
    bands=(0, 1, 2)      # RGB
):
    """
    Salva una patch come PNG.
    """
    os.makedirs(out_dir, exist_ok=True)

    img = patch_tensor[bands].detach().cpu().numpy()  # [3,H,W]
    img = np.transpose(img, (1, 2, 0))                 # [H,W,3]

    # normalizzazione semplice
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img = (img * 255).astype(np.uint8)

    filename = f"{sits_id}_{patch_id}_{timestamp}.png"
    path = os.path.join(out_dir, filename)

    imageio.imwrite(path, img)
    
"""def ensemble_predict(instance, ensemble):
    preds = []
    probas = []

    for cfg in ensemble.values():
        proba = cfg["model"].predict_proba(instance)
        probas.append(proba)
        preds.append(int(np.argmax(proba)))

    # Majority voting
    values, counts = np.unique(preds, return_counts=True)
    majority_idx = np.argmax(counts)

    if counts[majority_idx] >= (len(preds) / 2):
        return int(values[majority_idx])

    # Fallback: max probability globale
    probas = np.stack(probas)  # (N_models, N_classes)
    return int(np.argmax(probas.max(axis=0)))
"""


def ensemble_predict(instance, ensemble):
    preds = []
    probas = []
    confidences = []

    # Raccolta predizioni e probabilità
    for cfg in ensemble.values():
        proba = cfg["model"].predict_proba(instance)
        probas.append(proba)

        pred = int(np.argmax(proba))
        preds.append(pred)

        confidences.append(np.max(proba))

    confidences = np.array(confidences)

    # Seleziona i learner con confidenza massima
    max_conf = confidences.max()
    top_idx = np.where(confidences == max_conf)[0]

    top_preds = [preds[i] for i in top_idx]

    # Majority voting solo sui top-confidence learner
    values, counts = np.unique(top_preds, return_counts=True)
    majority_idx = np.argmax(counts)

    if counts[majority_idx] >= (len(top_preds) / 2):
        return int(values[majority_idx])

    # Fallback: classe con probabilità media più alta tra i top learner
    top_probas = np.stack([probas[i] for i in top_idx])  # (N_top, N_classes)
    return int(np.argmax(top_probas.mean(axis=0)))

# ---------------- Ensemble prediction with confidence-based selection ----------------
def ensemble_predict_confidence(instance, ensemble):
    best_adapt = {"conf": -1.0, "pred": None}
    best_frozen = {"conf": -1.0, "pred": None}

    for cfg in ensemble.values():
        proba = cfg["model"].predict_proba(instance)
        conf = float(np.max(proba))
        pred = int(np.argmax(proba))

        if cfg["adapt"]:
            if conf > best_adapt["conf"]:
                best_adapt = {"conf": conf, "pred": pred}
        else:
            if conf > best_frozen["conf"]:
                best_frozen = {"conf": conf, "pred": pred}

    # Fallback di sicurezza
    if best_adapt["pred"] is None:
        return best_frozen["pred"]
    if best_frozen["pred"] is None:
        return best_adapt["pred"]

    return (
        best_adapt["pred"]
        if best_adapt["conf"] > best_frozen["conf"]
        else best_frozen["pred"]
    )

# ---------------- Ensemble prediction with agreement-based selection ----------------
def ensemble_predict_agreement(instance, ensemble):
    adapt_preds, adapt_confs = [], []
    frozen_preds, frozen_confs = [], []

    for cfg in ensemble.values():
        proba = cfg["model"].predict_proba(instance)
        pred = int(np.argmax(proba))
        conf = float(np.max(proba))

        if cfg["adapt"]:
            adapt_preds.append(pred)
            adapt_confs.append(conf)
        else:
            frozen_preds.append(pred)
            frozen_confs.append(conf)

    # Majority vote interno
    adapt_vote = Counter(adapt_preds).most_common(1)[0][0] if adapt_preds else None
    frozen_vote = Counter(frozen_preds).most_common(1)[0][0] if frozen_preds else None

    # Caso semplice
    if adapt_vote is None:
        return frozen_vote
    if frozen_vote is None:
        return adapt_vote

    # Agreement tra gruppi
    if adapt_vote == frozen_vote:
        return adapt_vote

    # Disagreement → confronto confidenza media
    adapt_conf_mean = np.mean(adapt_confs)
    frozen_conf_mean = np.mean(frozen_confs)

    return adapt_vote if adapt_conf_mean > frozen_conf_mean else frozen_vote

# ---------------- Ensemble prediction with majority voting ----------------
def ensemble_predict_majority(instance, ensemble):
    votes = []
    best_conf = -1.0
    best_pred = None

    for cfg in ensemble.values():
        proba = cfg["model"].predict_proba(instance)
        pred = int(np.argmax(proba))
        conf = float(np.max(proba))

        votes.append(pred)

        # fallback: best single confidence
        if conf > best_conf:
            best_conf = conf
            best_pred = pred

    # Majority voting
    winner, count = Counter(votes).most_common(1)[0]

    if count >= len(votes) / 2:
        return winner

    return best_pred

# ---------------- Ensemble prediction with top-confidence voting ----------------
def ensemble_predict_top_confidence(instance, ensemble):
    preds = []
    confs = []
    probas = []

    for cfg in ensemble.values():
        proba = cfg["model"].predict_proba(instance)
        preds.append(int(np.argmax(proba)))
        confs.append(float(np.max(proba)))
        probas.append(proba)

    confs = np.array(confs)
    max_conf = confs.max()

    # Indici dei learner top-confidence
    top_idx = [i for i, c in enumerate(confs) if c == max_conf]
    top_preds = [preds[i] for i in top_idx]

    # Majority voting sui top learner
    winner, count = Counter(top_preds).most_common(1)[0]

    if count >= len(top_preds) / 2:
        return winner

    # Fallback: probabilità media solo sui top learner
    mean_proba = np.mean([probas[i] for i in top_idx], axis=0)
    return int(np.argmax(mean_proba))