import torch
from torchmetrics import Metric
import numpy as np


class SCDMetricPatch(Metric):
    """
    Patch-level version of SCDMetric.
    Computes mIoU, per-class IoU, BC, SC, SCS, and accuracy metrics on patch-level predictions.
    
    Args:
        num_classes (int): number of semantic classes.
        class_names (list): list of class names.
        ignore_index (int, optional): index to ignore during metric computation.
    """

    def __init__(self, num_classes, class_names, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names
        self.ignore_index = ignore_index

        self.conf_matrix = np.zeros((num_classes, num_classes))
        self.conf_matrix_change = np.zeros((2, 2))
        self.conf_matrix_sc = np.zeros((num_classes, num_classes))

    def update(self, pred, gt):
        """
        Update confusion matrices based on patch-level predictions.
        Args:
            pred: [B, T] tensor of predicted class indices
            gt: [B, T] tensor of ground truth class indices
        """
        assert pred.shape == gt.shape, f"Shape mismatch: pred={pred.shape}, gt={gt.shape}"

        # Flatten batch/time for per-sample computation
        gt = gt.long().T.reshape(gt.shape[1], -1)    # T x N
        pred = pred.long().T.reshape(pred.shape[1], -1)  # T x N

        # Binary change detection (change between consecutive timesteps)
        gt_change = (gt[1:] != gt[:-1]).int()  # (T-1) x N
        pred_change = (pred[1:] != pred[:-1]).int()  # (T-1) x N

        # Handle ignore index if defined
        if self.ignore_index is not None:
            udm_mask = (gt == self.ignore_index).int()
            mixed_udm_mask = 1 - torch.clamp(udm_mask[1:] + udm_mask[:-1], 0, 1)
            udm_mask = 1 - udm_mask
        else:
            udm_mask = torch.ones_like(gt).int()
            mixed_udm_mask = torch.ones_like(gt_change).int()

        gt_np = gt.cpu().numpy()
        pred_np = pred.cpu().numpy()
        gt_change_np = gt_change.cpu().numpy()
        pred_change_np = pred_change.cpu().numpy()
        udm_mask_np = udm_mask.cpu().numpy().flatten()
        mixed_udm_mask_np = mixed_udm_mask.cpu().numpy().flatten()

        mask = (gt_np.flatten() >= 0) & (gt_np.flatten() < self.num_classes) & (udm_mask_np == 1)
        mask_change = (gt_change_np.flatten() >= 0) & (gt_change_np.flatten() < 2) & (mixed_udm_mask_np == 1)
        mask_semantic_change = (
            (gt_np[1:].flatten() >= 0)
            & (gt_np[1:].flatten() < self.num_classes)
            & (mixed_udm_mask_np == 1)
            & (gt_change_np.flatten() == 1)
        )

        # Update confusion matrices
        self.conf_matrix += np.bincount(
            self.num_classes * gt_np.flatten()[mask].astype(int) + pred_np.flatten()[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)

        self.conf_matrix_change += np.bincount(
            2 * gt_change_np.flatten()[mask_change].astype(int) + pred_change_np.flatten()[mask_change],
            minlength=2 ** 2,
        ).reshape(2, 2)

        self.conf_matrix_sc += np.bincount(
            self.num_classes * gt_np[1:].flatten()[mask_semantic_change].astype(int)
            + pred_np[1:].flatten()[mask_semantic_change],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)

    def compute(self):
        conf_mat = self.conf_matrix
        conf_mat_change = self.conf_matrix_change
        conf_mat_sc = self.conf_matrix_sc

        # mIoU and per-class IoU
        miou, per_class_iou, conf_mat_iou = compute_miou(conf_mat)

        # Binary Change (BC)
        bc = np.divide(
            conf_mat_change[1, 1],
            conf_mat_change.sum() - conf_mat_change[0, 0],
            out=np.zeros_like(conf_mat_change[1, 1]),
            where=conf_mat_change.sum() - conf_mat_change[0, 0] != 0,
        ) * 100

        # Semantic Change (SC)
        sc, _, _ = compute_miou(conf_mat_sc)
        scs = 0.5 * (bc + sc)

        # Mean accuracy
        macc = np.nanmean(
            np.divide(
                np.diag(conf_mat),
                np.sum(conf_mat, axis=1),
                out=np.zeros_like(np.diag(conf_mat)),
                where=np.sum(conf_mat, axis=1) != 0,
            )
        ) * 100

        # Precision, recall, F1
        per_class_precision = np.divide(
            np.diag(conf_mat),
            np.sum(conf_mat, axis=0),
            out=np.zeros_like(np.diag(conf_mat), dtype=float),
            where=np.sum(conf_mat, axis=0) != 0,
        )

        per_class_recall = np.divide(
            np.diag(conf_mat),
            np.sum(conf_mat, axis=1),
            out=np.zeros_like(np.diag(conf_mat), dtype=float),
            where=np.sum(conf_mat, axis=1) != 0,
        )

        per_class_f1 = np.divide(
            2 * per_class_precision * per_class_recall,
            per_class_precision + per_class_recall + 1e-8,
            out=np.zeros_like(per_class_precision, dtype=float),
            where=(per_class_precision + per_class_recall) != 0,
        )

        # Macro averages
        precision_macro = np.nanmean(per_class_precision) * 100
        recall_macro = np.nanmean(per_class_recall) * 100
        f1_macro = np.nanmean(per_class_f1) * 100

        # Confusion matrix normalized (percent)
        conf_mat_percent = np.divide(
            conf_mat, conf_mat.sum(axis=1, keepdims=True),
            out=np.zeros_like(conf_mat, dtype=float),
            where=conf_mat.sum(axis=1, keepdims=True) != 0
        ) * 100

        conf_mat_change_percent = np.divide(
            conf_mat_change, conf_mat_change.sum(axis=1, keepdims=True),
            out=np.zeros_like(conf_mat_change, dtype=float),
            where=conf_mat_change.sum(axis=1, keepdims=True) != 0
        ) * 100

        conf_mat_sc_percent = np.divide(
            conf_mat_sc, conf_mat_sc.sum(axis=1, keepdims=True),
            out=np.zeros_like(conf_mat_sc, dtype=float),
            where=conf_mat_sc.sum(axis=1, keepdims=True) != 0
        ) * 100

        output = {
            "acc": np.diag(conf_mat).sum() / conf_mat.sum() * 100,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "macc": macc,
            "miou": miou,
            "bc": bc,
            "sc": sc,
            "scs": scs,
            "confusion_matrix": conf_mat_percent,
            "confusion_matrix_change": conf_mat_change_percent,
            "confusion_matrix_sc": conf_mat_sc_percent,
            "confusion_matrix_iou": conf_mat_iou
        }

        for class_id, class_name in enumerate(self.class_names):
            output[class_name] = per_class_iou[class_id]
            output[f"{class_name}_precision"] = per_class_precision[class_id] * 100
            output[f"{class_name}_recall"] = per_class_recall[class_id] * 100
            output[f"{class_name}_f1"] = per_class_f1[class_id] * 100

        # Reset for next epoch
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes))
        self.conf_matrix_change = np.zeros((2, 2))
        self.conf_matrix_sc = np.zeros((self.num_classes, self.num_classes))

        return output


def compute_miou(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    iou_matrix = np.zeros_like(confusion_matrix, dtype=float)
    per_class_iou = np.zeros(num_classes, dtype=float)

    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fn = np.sum(confusion_matrix[i, :]) - tp
        fp = np.sum(confusion_matrix[:, i]) - tp
        denom = tp + fn + fp
        if denom > 0:
            per_class_iou[i] = tp / denom
            iou_matrix[i, i] = per_class_iou[i]
            for j in range(num_classes):
                if j != i:
                    iou_matrix[i, j] = (confusion_matrix[i, j] + confusion_matrix[j, i]) / denom

    miou = np.nanmean(per_class_iou) * 100
    return miou, per_class_iou * 100, iou_matrix * 100