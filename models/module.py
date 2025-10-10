import pytorch_lightning as L
import torch
import torch.nn as nn
from hydra.utils import instantiate
from pathlib import Path
import numpy as np
import wandb
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from torchmetrics import Metric


class SitsScdModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = instantiate(cfg.network.instance)
        self.loss = instantiate(cfg.loss.instance)
        self.ignore_index = self.loss.ignore_index
        self.val_metrics = instantiate(cfg.val_metrics)
        self.test_metrics = instantiate(cfg.test_metrics)
        self.dataset = self.cfg.dataset.name
        self.global_batch_size = self.cfg.dataset.global_batch_size
        self.logged_val_images = False
        self.class_distribution = ClassDistribution(num_classes=len(CLASS_NAMES), ignore_index=self.ignore_index)

    def training_step(self, batch, batch_idx):
        pred = self.model(batch)
        logits = pred["logits"]  # B x T x C x H x W
        logits = logits.mean(dim=[-2,-1])  # B x T x C
        pred["pred"] = torch.argmax(logits, dim=2) # B x T
        gt = batch["gt"]  # B x T x H x W
        gt = gt.float().mean(dim=[-2,-1]).round().long()
        loss = self.loss(logits, gt, average=True)
        for metric_name, metric_value in loss.items():
            self.log(
                f"train/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )

        # Log images at specified intervals
        if batch_idx % self.cfg.logging.train_image_interval == 0 and self.global_rank == 0:
            pred["pred"] = torch.argmax(pred["logits"], dim=2)
            self.log_wandb_images(pred["pred"], batch["gt"], batch_idx, batch["data"], prefix="train", dataset_type=self.dataset, max_samples=self.global_batch_size)
        
        freqs = self.class_distribution.compute()
        for cls_id, freq in enumerate(freqs):
            if cls_id != self.ignore_index:
                    self.log(f"train_freq/{CLASS_NAMES[cls_id]}", freq.item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.class_distribution.reset()
        
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pred = self.model(batch)
        logits = pred["logits"]  # B x T x C x H x W
        logits = logits.mean(dim=[-2,-1])  # B x T x C
        pred["pred"] = torch.argmax(logits, dim=2) # B x T
        gt = batch["gt"]  # B x T x H x W
        gt = gt.float().mean(dim=[-2,-1]).round().long()  # B x T
        loss = self.loss(pred, batch, average=True)["loss"]
        self.val_metrics.update(pred["pred"], gt)
        self.log("val/loss", loss, sync_dist=True, on_step=False, on_epoch=True)
        
        with torch.no_grad() :
            values, counts = torch.unique(batch["gt"], return_counts=True)
            total_pixels = counts.sum().item()
            freqs = {int(v): (c.item() / total_pixels) * 100 for v, c in zip(values, counts) if v != self.ignore_index}
            for cls_id, freq in freqs.items():
                class_name = CLASS_NAMES[cls_id]
                self.log(f"val_freq/{class_name}", freq, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
                        
        # Log images at specified intervals
        if (not self.logged_val_images) and (batch_idx % self.cfg.logging.val_image_interval == 0) and self.global_rank == 0:
            self.log_wandb_images(pred["pred"], batch["gt"], batch_idx, batch["data"], prefix="val", dataset_type=self.dataset, max_samples=self.global_batch_size)
    
    def on_validation_epoch_end(self):
        computed = self.val_metrics.compute()
        self.log_metrics(computed, prefix="val")
        self.logged_val_images = True  # Only log once per epoch

        if self.global_rank == 0:
            # Full-class confusion matrix
            if "confusion_matrix" in computed:
                cm = computed["confusion_matrix"]
                fig = plot_confusion_matrix(cm, self.val_metrics.class_names, title="Validation Confusion Matrix Pixel Classification")
                wandb.log({
                    "val_matrix/confusion_matrix_pixel_classification": wandb.Image(fig)
                })
                plt.close(fig)

            # Binary change detection confusion matrix
            if "confusion_matrix_change" in computed:
                cm_change = computed["confusion_matrix_change"]
                fig_change = plot_confusion_matrix(cm_change, ["No Change", "Change"], title="Validation Change Confusion Matrix Pixel Classification")
                wandb.log({
                    "val_matrix/confusion_matrix_change_pixel_classification": wandb.Image(fig_change),
                })
                plt.close(fig_change)

            # Semantic change confusion matrix
            if "confusion_matrix_sc" in computed:
                cm_sc = computed["confusion_matrix_sc"]
                fig_sc = plot_confusion_matrix(cm_sc, self.val_metrics.class_names, title="Validation Semantic Change Confusion Matrix Pixel Classification")
                wandb.log({
                    "val_matrix/confusion_matrix_sc_pixel_classification": wandb.Image(fig_sc),
                })
                plt.close(fig_sc)
            if "confusion_matrix_iou" in computed:
                cm_iou = computed["confusion_matrix_iou"]
                fig_iou = plot_confusion_matrix(cm_iou, self.val_metrics.class_names, title="Validation IoU Confusion Matrix")
                wandb.log({
                    "val_matrix/confusion_matrix_iou": wandb.Image(fig_iou),
                })
                plt.close(fig_iou)
        
        freqs = self.class_distribution.compute()
        for cls_id, freq in enumerate(freqs):
            if cls_id != self.ignore_index:
                self.log(f"val_freq/{CLASS_NAMES[cls_id]}", freq.item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.class_distribution.reset()

        self.val_metrics.reset()
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pred = self.model(batch)
        logits = pred["logits"]  # B x T x C x H x W
        logits = logits.mean(dim=[-2,-1])  # B x T x C
        pred["pred"] = torch.argmax(logits, dim=2) # B x T

        # Enable to save predictions local
        #self.save_predictions(pred["pred"], batch_idx)
        gt = batch["gt"]  # B x T x H x W
        gt = gt.float().mean(dim=[-2,-1]).round().long()  # B x T
        self.test_metrics.update(pred["pred"], gt)
        self.class_distribution.update(batch["gt"])
        
        self.log_wandb_images(pred["pred"], batch["gt"], batch_idx, batch["data"], prefix="test", dataset_type=self.dataset, max_samples=self.global_batch_size)
    
    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        
        # Log scalar metrics
        self.log_metrics(metrics, prefix="test")

        if self.global_rank == 0:
            # Confusion matrices: log as images and tables
            if "confusion_matrix" in metrics:
                cm = metrics["confusion_matrix"]
                fig = plot_confusion_matrix(cm, self.test_metrics.class_names, title="Test Confusion Matrix Pixel Classification")
                wandb.log({
                    "test_matrix/confusion_matrix_pixel_classification": wandb.Image(fig),
                })
                plt.close(fig)

            if "confusion_matrix_change" in metrics:
                cm_change = metrics["confusion_matrix_change"]
                fig_change = plot_confusion_matrix(cm_change, ["No Change", "Change"], title="Test Change Confusion Matrix Pixel Classification")
                wandb.log({
                    "test_matrix/confusion_matrix_change_pixel_classification": wandb.Image(fig_change),
                })
                plt.close(fig_change)

            if "confusion_matrix_sc" in metrics:
                cm_sc = metrics["confusion_matrix_sc"]
                fig_sc = plot_confusion_matrix(cm_sc, self.test_metrics.class_names, title="Test Semantic Change Confusion Matrix Pixel Classification")
                wandb.log({
                    "test_matrix/confusion_matrix_sc_pixel_classification": wandb.Image(fig_sc),
                })
                plt.close(fig_sc)
                
            if "confusion_matrix_iou" in metrics:
                cm_iou = metrics["confusion_matrix_iou"]
                fig_iou = plot_confusion_matrix(cm_iou, self.test_metrics.class_names, title="Test IoU Confusion Matrix")
                wandb.log({
                    "test_matrix/confusion_matrix_iou": wandb.Image(fig_iou),
                })
                plt.close(fig_iou)
        
        freqs = self.class_distribution.compute()
        for cls_id, freq in enumerate(freqs):
            if cls_id != self.ignore_index:
                self.log(f"test_freq/{CLASS_NAMES[cls_id]}", freq.item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.class_distribution.reset()

        self.test_metrics.reset()


    def configure_optimizers(self):
        if self.cfg.optimizer.exclude_ln_and_biases_from_weight_decay:
            params_with_wd, params_without_wd = self.split_weight_decay_params()
            param_groups = [
                {"params": params_with_wd, "weight_decay": self.cfg.optimizer.optim.weight_decay},
                {"params": params_without_wd, "weight_decay": 0.0},
            ]
            optimizer = instantiate(self.cfg.optimizer.optim, param_groups)
        else:
            optimizer = instantiate(self.cfg.optimizer.optim, self.model.parameters())

        scheduler = instantiate(self.cfg.lr_scheduler)(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.global_step)

    def split_weight_decay_params(self):
        param_names = get_parameter_names(self.model, [nn.LayerNorm])
        params_with_wd = [p for n, p in self.model.named_parameters() if n in param_names and "bias" not in n]
        params_without_wd = [p for n, p in self.model.named_parameters() if n not in param_names or "bias" in n]
        return params_with_wd, params_without_wd
            
    def save_predictions(self, preds, batch_idx):
        output_dir = Path(self.cfg.output_dir) / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        for i in range(preds.shape[0]):
            np.save(output_dir / f"sample_{batch_idx}_{i}.npy", preds[i].cpu().numpy())

    def log_metrics(self, metrics, prefix):
      for name, value in metrics.items():
          # Only log scalars with self.log
          if isinstance(value, (int, float, torch.Tensor, np.floating, np.integer)):
              self.log(f"{prefix}/{name}", value, sync_dist=True, on_step=False, on_epoch=True)
          else:
              # Skip non-scalars (like confusion matrices) 
              print(f"Skipping {prefix}/{name} from self.log() because it is type {type(value)}")

    def log_wandb_images(self, preds, gt, batch_idx, data, prefix="test", dataset_type="Muds", max_samples=4):
        B = preds.shape[0]
        num_samples = min(B, max_samples)

        for b in range(num_samples):
            preds_np = preds[b].cpu().numpy() if prefix in ["val", "test"] else None
            gt_np = gt[b].cpu().numpy() if (gt is not None and len(gt) > 0) else None
            input_np = data[b].cpu().numpy()  # [T, C, H, W]

            # For DynamicEarthNet: split RGB and IR channels
            if dataset_type == "DynamicEarthNet":
                input_rgb = input_np[:, :3, :, :]  # [T, 3, H, W]
                input_ir = input_np[:, 3:, :, :]   # [T, 1, H, W]
            else:
                input_rgb = input_np  # MUDS is già RGB
                input_ir = None

            # Helper: normalize image
            def normalize_img(img):
                img = np.moveaxis(img, 0, -1)  # C,H,W → H,W,C
                img_vis = (img - img.min()) / (img.max() - img.min() + 1e-5)
                return (img_vis * 255).astype(np.uint8)

            # Format image (colormap)
            def format_image(img_array):
                if dataset_type == "Muds":
                    return to_binary_colormap_image(img_array)
                elif dataset_type == "DynamicEarthNet":
                    return to_class_colormap_image(img_array)
                return None

            # --- Log RGB input ---
            input_images = {
                f"{prefix}_input/t{t:02d}": wandb.Image(normalize_img(input_rgb[t]), caption=f"Batch {batch_idx} Sample {b}")
                for t in range(input_rgb.shape[0])
            }
            wandb.log(input_images)

            # --- Log IR input (solo per DynamicEarthNet) ---
            if input_ir is not None:
                input_ir_images = {
                    f"{prefix}_input_infrared/t{t:02d}": wandb.Image(normalize_img(np.repeat(input_ir[t], 3, axis=0)),
                                                                            caption=f"Batch {batch_idx} Sample {b}")
                    for t in range(input_ir.shape[0])
                }
                wandb.log(input_ir_images)

            # --- Log predictions ---
            if preds_np is not None:            
                pred_images = {
                    f"{prefix}_pred/t{t:02d}": wandb.Image(format_image(preds_np[t]), caption=f"Batch {batch_idx} Sample {b}")
                    for t in range(preds_np.shape[0])
                }
                wandb.log(pred_images)

            # --- Log ground truth ---
            if gt_np is not None:
                gt_images = {}
                for t in range(gt_np.shape[0]):
                    img_fmt = format_image(gt_np[t])
                    if img_fmt is not None:
                        gt_images[f"{prefix}_gt/t{t:02d}"] = wandb.Image(
                            img_fmt, caption=f"Batch {batch_idx} Sample {b}"
                        )
                if gt_images:
                    wandb.log(gt_images)

            # --- Log temporal stats ---
            """temporal_images = {
                f"{prefix}_temporal/sample{b}_mean": wandb.Image(format_image(np.mean(preds_np, axis=0)), caption=f"Batch {batch_idx} Sample {b}"),
                f"{prefix}_temporal/sample{b}_std": wandb.Image(format_image(np.std(preds_np, axis=0)), caption=f"Batch {batch_idx} Sample {b}"),
                f"{prefix}_temporal/sample{b}_change": wandb.Image(format_image(preds_np[-1] - preds_np[0]), caption=f"Batch {batch_idx} Sample {b}")
            }
            wandb.log(temporal_images) """


class ClassDistribution(Metric):
    def __init__(self, num_classes: int, ignore_index: int = None, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # accumulator for counts
        self.add_state("counts", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, gt: torch.Tensor):
        """
        gt: Tensor of shape (B x T x H x W)
        """
        gt_flat = gt.view(-1)

        if self.ignore_index is not None:
            mask = gt_flat != self.ignore_index
            gt_flat = gt_flat[mask]

        values, counts = torch.unique(gt_flat, return_counts=True)
        for v, c in zip(values, counts):
            self.counts[v] += c

    def compute(self):
        total = self.counts.sum().item()
        freqs = (self.counts.float() / total) * 100 if total > 0 else torch.zeros_like(self.counts, dtype=torch.float)
        return freqs

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result


def to_binary_colormap_image(array, figsize=(2.56, 2.56), dpi=100):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis("off")
    ax.imshow(array, cmap="binary")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

CLASS_NAMES = ["impervi", "agricult", "forest", "wetlands", "soil", "water", "unknown"]
CLASS_COLORS = np.array([
    [64, 64, 64],  # impervi (gray)
    [204, 204, 0],    # agricult (yellow)
    [0, 204, 0],      # forest (green)
    [0, 0, 102],      # wetlands (blue)
    [153, 76, 0],    # soil (brown)
    [51, 51, 255],  # water (light blue)
    [0, 0, 0]   # unknown (black)
], dtype=np.uint8)

def to_class_colormap_image(img_array):
    """
    Convert a 2D or 3D numpy array of class indices to a colored image using predefined class colors.
    """
    if img_array.ndim == 3 and img_array.shape[0] == 1:  
        img_array = img_array.squeeze(0)  # (1, H, W) → (H, W)
    colored = CLASS_COLORS[img_array.astype(int)]
    return colored

def confusion_matrix_to_wandb_table(cm, class_names):
    """
    Convert a confusion matrix (numpy array) into a wandb.Table
    for interactive visualization.
    """
    table = wandb.Table(columns=["Actual", "Predicted", "Count"])
    n_classes = cm.shape[0]
    for i in range(n_classes):
        for j in range(n_classes):
            table.add_data(class_names[i], class_names[j], int(cm[i, j]))
    return table

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))

    # Convert to numpy array if it's a tensor
    if hasattr(cm, 'cpu'):
        cm = cm.cpu().numpy()
    
    # Always use float formatting to avoid format errors
    # This works for both int and float values
    fmt = ".3f"

    #print(f"CM shape: {cm.shape}, dtype: {cm.dtype}, sample values: {cm.flat[:5]}")
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    fig = plt.gcf()
    return fig