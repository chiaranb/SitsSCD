import torch
from torchmetrics import Metric
import numpy as np


class SCDMetric(Metric):
    """
    Computes the mean intersection-over-union (miou), the binary change score (bc), the semantic change score (sc)
    and the semantic change segmentation score (scs). Additionally provides the accuracy (acc) and mean accuracy (macc)
    for semantic segmentation.

    Args:
        num_classes (int): the number of semantic classes.
        ignore_index (int): ground truth index to ignore in the metrics.
    """

    def __init__(self, num_classes, class_names, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names
        self.ignore_index = ignore_index
        # Initializes three confusion matrices:
        # 1. For semantic segmentation (num_classes x num_classes)
        # 2. For binary change detection (2 x 2)
        # 3. For semantic change segmentation (num_classes x num_classes)
        self.conf_matrix = np.zeros((num_classes, num_classes))
        self.conf_matrix_change = np.zeros((2, 2))
        self.conf_matrix_sc = np.zeros((num_classes, num_classes))

    def update(self, pred, gt):
        """
        Update the confusion matrices
        :param pred: B x T x H x W (Batch x Time x Height x Width)
        :param gt: B x T x H x W (Batch x Time x Height x Width)
        :return: None
        """
        gt = gt.permute(1, 0, 2, 3).reshape(gt.shape[1], -1).long()  # T x N
        pred = pred.long().permute(1, 0, 2, 3).reshape(pred.shape[1], -1)  # T x N

        gt_change = (gt[1:] != gt[:-1]).int()  # (T-1) x N
        pred_change = (pred[1:] != pred[:-1]).int()  # (T-1) x N, 1 if change, 0 otherwise

        udm_mask = (gt == self.ignore_index).int()  # undefined data mask
        mixed_udm_mask = 1 - torch.clamp(udm_mask[1:] + udm_mask[:-1], 0, 1)
        udm_mask = 1 - udm_mask

        gt = gt.cpu().numpy()
        pred = pred.cpu().numpy()
        udm_mask = udm_mask.flatten().cpu().numpy()
        mixed_udm_mask = mixed_udm_mask.flatten().cpu().numpy()
        gt_change = gt_change.flatten().cpu().numpy()
        pred_change = pred_change.flatten().cpu().numpy()

        mask = (gt.flatten() >= 0) & (gt.flatten() < self.num_classes) & (udm_mask == 1)
        mask_change = (gt_change >= 0) & (gt_change < 2) & (mixed_udm_mask == 1)
        mask_semantic_change = (gt[1:].flatten() >= 0) & (gt[1:].flatten() < self.num_classes) & (mixed_udm_mask == 1) & (gt_change == 1)

        # Update confusion matrices
        self.conf_matrix += np.bincount(self.num_classes * gt.flatten()[mask].astype(int) + pred.flatten()[mask],
                                        minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

        self.conf_matrix_change += np.bincount(2 * gt_change[mask_change].astype(int) + pred_change[mask_change],
                                               minlength=2 ** 2).reshape(2, 2)

        self.conf_matrix_sc += np.bincount(self.num_classes * gt[1:].flatten()[mask_semantic_change].astype(int) + pred[1:].flatten()[mask_semantic_change],
                                           minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

    def compute(self):
        conf_mat = self.conf_matrix
        conf_mat_change = self.conf_matrix_change
        conf_mat_sc = self.conf_matrix_sc
        
        # Mean Intersection over Union (mIoU) and per-class IoU
        miou, per_class_iou = compute_miou(conf_mat)
        
        # Binary Change Score (bc), Semantic Change Score (sc), and Semantic Change Segmentation Score (scs)
        sc, _ = compute_miou(conf_mat_sc)
        bc = np.divide(conf_mat_change[1, 1], conf_mat_change.sum() - conf_mat_change[0, 0],
                       out=np.zeros_like(conf_mat_change[1, 1]),
                       where=conf_mat_change.sum() - conf_mat_change[0, 0] != 0) * 100
        macc = np.nanmean(np.divide(np.diag(conf_mat), np.sum(conf_mat, axis=1),
                       out=np.zeros_like(np.diag(conf_mat)),
                       where=np.sum(conf_mat, axis=1) != 0)) * 100
        scs = 0.5 * (bc + sc)
        # Per-class precision, recall, f1
        per_class_precision = np.divide(np.diag(conf_mat), np.sum(conf_mat, axis=0),
                                        out=np.zeros_like(np.diag(conf_mat), dtype=float),
                                        where=np.sum(conf_mat, axis=0) != 0)

        per_class_recall = np.divide(np.diag(conf_mat), np.sum(conf_mat, axis=1),
                                    out=np.zeros_like(np.diag(conf_mat), dtype=float),
                                    where=np.sum(conf_mat, axis=1) != 0)

        per_class_f1 = np.divide(2 * per_class_precision * per_class_recall,
                                per_class_precision + per_class_recall + 1e-8,
                                out=np.zeros_like(per_class_precision, dtype=float),
                                where=(per_class_precision + per_class_recall) != 0)

        # Macro averages
        precision_macro = np.nanmean(per_class_precision) * 100
        recall_macro = np.nanmean(per_class_recall) * 100
        f1_macro = np.nanmean(per_class_f1) * 100
        
        conf_mat_percent = (conf_mat / conf_mat.sum()) * 100
        conf_mat_change_percent = (conf_mat_change / conf_mat_change.sum()) * 100
        conf_mat_sc_percent = (conf_mat_sc / conf_mat_sc.sum()) * 100
        
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
            "confusion_matrix_sc": conf_mat_sc_percent
        }
        
        for class_id, class_name in enumerate(self.class_names):
            output[class_name] = per_class_iou[class_id]
            output[f"{class_name}_precision"] = per_class_precision[class_id] * 100
            output[f"{class_name}_recall"] = per_class_recall[class_id] * 100
            output[f"{class_name}_f1"] = per_class_f1[class_id] * 100
            
        # Reset confusion matrices for the next computation
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes))
        self.conf_matrix_change = np.zeros((2, 2))
        self.conf_matrix_sc = np.zeros((self.num_classes, self.num_classes))
        return output

# Computes the mean intersection-over-union (mIoU) and per-class IoU from a confusion matrix.
def compute_miou(confusion_matrix):
    den_iou = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    per_class_iou = np.divide(np.diag(confusion_matrix), den_iou, out=np.zeros_like(den_iou), where=den_iou != 0)
    return np.nanmean(per_class_iou) * 100, per_class_iou * 100