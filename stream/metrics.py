import numpy as np

# Costanti condivise
CLASS_NAMES = ["impervi", "agricult", "forest", "wetlands", "soil", "water"]
NUM_CLASSES = len(CLASS_NAMES)

class StreamingChangeEvaluator:
    """
    Valuta la classificazione e il rilevamento dei cambiamenti in uno stream
    ordinato per tempo (t0_all, t1_all, ...).
    
    Mantiene uno stato (t-1) per ogni patch_id.
    Calcola solo mIoU e le metriche di cambiamento (BC, SC, SCS).
    """
    def __init__(self, num_classes, ignore_index=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        self._last_state = {}
        
        self.conf_matrix = np.zeros((num_classes, num_classes))
        self.conf_matrix_change = np.zeros((2, 2))
        self.conf_matrix_sc = np.zeros((num_classes, num_classes))

    def update(self, patch_id, y_true, y_pred):
        """Aggiorna le metriche con la nuova istanza (y_true, y_pred) per un dato patch_id."""
        
        y_true, y_pred = int(y_true), int(y_pred)
        
        if y_true == self.ignore_index:
            return 

        self.conf_matrix[y_true, y_pred] += 1
        
        if patch_id in self._last_state:
            y_true_t_minus_1, y_pred_t_minus_1 = self._last_state[patch_id]

            gt_change = 1 if y_true != y_true_t_minus_1 else 0
            pred_change = 1 if y_pred != y_pred_t_minus_1 else 0

            self.conf_matrix_change[gt_change, pred_change] += 1

            if gt_change == 1:
                self.conf_matrix_sc[y_true, y_pred] += 1

        self._last_state[patch_id] = (y_true, y_pred)

    def compute(self):
            """Calcola e restituisce le metriche di mIoU e cambiamento."""
            
            conf_mat = self.conf_matrix
            conf_mat_change = self.conf_matrix_change
            conf_mat_sc = self.conf_matrix_sc

            # --- Calcolo mIoU (da conf_mat) ---
            tp = np.diag(conf_mat)
            support = conf_mat.sum(axis=1)
            pred_counts = conf_mat.sum(axis=0)
            fp = pred_counts - tp
            fn = support - tp
            
            iou_denom = tp + fp + fn
            iou = tp / (iou_denom + 1e-8)
            miou = np.nanmean(iou[support > 0]) * 100
            if np.isnan(miou): miou = 0.0

            # --- Calcoli di Cambiamento ---
            
            # Binary Change (BC)
            bc_denom = conf_mat_change.sum() - conf_mat_change[0, 0]
            bc = (conf_mat_change[1, 1] / (bc_denom + 1e-8)) * 100

            # Semantic Change (SC) - mIoU sulla matrice SC
            tp_sc = np.diag(conf_mat_sc)
            support_sc = conf_mat_sc.sum(axis=1)
            fp_sc = conf_mat_sc.sum(axis=0) - tp_sc
            fn_sc = support_sc - tp_sc
            iou_sc_denom = tp_sc + fp_sc + fn_sc
            iou_sc = tp_sc / (iou_sc_denom + 1e-8)
            sc = np.nanmean(iou_sc[support_sc > 0]) * 100
            if np.isnan(sc): sc = 0.0

            # SCS
            scs = 0.5 * (bc + sc)

            return {
                "miou": miou,
                "bc": bc,
                "sc": sc,
                "scs": scs,
            }