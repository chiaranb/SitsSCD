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

    def training_step(self, batch, batch_idx):
        pred = self.model(batch)
        loss = self.loss(pred, batch, average=True)
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
            self.log_wandb_images(pred["pred"], batch["gt"], batch_idx, batch["data"], prefix="train", dataset_type=self.dataset)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pred = self.model(batch)
        pred["pred"] = torch.argmax(pred["logits"], dim=2)
        loss = self.loss(pred, batch, average=True)["loss"]
        self.val_metrics.update(pred["pred"], batch["gt"])
        self.log("val/loss", loss, sync_dist=True, on_step=False, on_epoch=True)
        
        # Log images at specified intervals
        if batch_idx % self.cfg.logging.val_image_interval == 0 and self.global_rank == 0:
            self.log_wandb_images(pred["pred"], batch["gt"], batch_idx, batch["data"], prefix="val", dataset_type=self.dataset)
    
    def on_validation_epoch_end(self):
        computed = self.val_metrics.compute()
        self.log_metrics(computed, prefix="val")

        if self.global_rank == 0:
            # Full-class confusion matrix
            if "confusion_matrix" in computed:
                cm = computed["confusion_matrix"]
                fig = plot_confusion_matrix(cm, self.val_metrics.class_names, title="Validation Confusion Matrix")
                wandb.log({"val/confusion_matrix_image": wandb.Image(fig)})
                plt.close(fig)
                
                table = confusion_matrix_to_wandb_table(cm, self.val_metrics.class_names)
                wandb.log({"val/confusion_matrix_table": table})

            # Binary change detection confusion matrix
            if "confusion_matrix_change" in computed:
                cm_change = computed["confusion_matrix_change"]
                fig_change = plot_confusion_matrix(cm_change, ["No Change", "Change"], title="Validation Change Confusion Matrix")
                wandb.log({"val/confusion_matrix_change_image": wandb.Image(fig_change)})
                plt.close(fig_change)

                table_change = confusion_matrix_to_wandb_table(cm_change, ["No Change", "Change"])
                wandb.log({"val/confusion_matrix_change_table": table_change})

            # Semantic change confusion matrix
            if "confusion_matrix_sc" in computed:
                cm_sc = computed["confusion_matrix_sc"]
                fig_sc = plot_confusion_matrix(cm_sc, self.val_metrics.class_names, title="Validation Semantic Change Confusion Matrix")
                wandb.log({"val/confusion_matrix_sc_image": wandb.Image(fig_sc)})
                plt.close(fig_sc)

                table_sc = confusion_matrix_to_wandb_table(cm_sc, self.val_metrics.class_names)
                wandb.log({"val/confusion_matrix_sc_table": table_sc})

        self.val_metrics.reset()
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pred = self.model(batch)
        pred["pred"] = torch.argmax(pred["logits"], dim=2)

        # Enable to save predictions local
        #self.save_predictions(pred["pred"], batch_idx)
        self.test_metrics.update(pred["pred"], batch["gt"])
        self.log_wandb_images(pred["pred"], batch["gt"], batch_idx, batch["data"], prefix="test", dataset_type=self.dataset)
    
    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log_metrics(metrics, prefix="test")

        if self.global_rank == 0:
            if "confusion_matrix" in metrics:
                cm = metrics["confusion_matrix"]
                wandb.log({
                    "test/confusion_matrix_image": wandb.Image(plot_confusion_matrix(cm, self.test_metrics.class_names, title="Test Confusion Matrix")),
                    "test/confusion_matrix_table": confusion_matrix_to_wandb_table(cm, self.test_metrics.class_names)
                })

            if "confusion_matrix_change" in metrics:
                cm_change = metrics["confusion_matrix_change"]
                wandb.log({
                    "test/confusion_matrix_change_image": wandb.Image(plot_confusion_matrix(cm_change, ["No Change", "Change"], title="Test Change Confusion Matrix")),
                    "test/confusion_matrix_change_table": confusion_matrix_to_wandb_table(cm_change, ["No Change", "Change"])
                })

            if "confusion_matrix_sc" in metrics:
                cm_sc = metrics["confusion_matrix_sc"]
                wandb.log({
                    "test/confusion_matrix_sc_image": wandb.Image(plot_confusion_matrix(cm_sc, self.test_metrics.class_names, title="Test Semantic Change Confusion Matrix")),
                    "test/confusion_matrix_sc_table": confusion_matrix_to_wandb_table(cm_sc, self.test_metrics.class_names)
                })

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
            self.log(f"{prefix}/{name}", value, sync_dist=True, on_step=False, on_epoch=True)

    def log_wandb_images(self, preds, gt, batch_idx, data, prefix="test", dataset_type="Muds"):
        # Extract numpy arrays
        preds_np = preds[0].cpu().numpy() if prefix in ("test", "val") else None
        gt_np = gt[0].cpu().numpy() if (gt is not None and len(gt) > 0) else None

        # First batch inputs
        input_np = data[0].cpu().numpy()  # shape [T, C, H, W]

        # For DynamicEarthNet: split RGB and IR channels
        input_rgb = None
        input_ir = None
        if dataset_type == "DynamicEarthNet":
            input_rgb = input_np[:, :3, :, :]  # [T, 3, H, W]
            input_ir = input_np[:, 3:, :, :]   # [T, 1, H, W]
        else:
            input_rgb = input_np  # MUDS is already RGB only

        # Helper function: normalize image to 0–255 uint8
        def normalize_img(img):
            img = np.moveaxis(img, 0, -1)  # C,H,W → H,W,C
            img_vis = (img - img.min()) / (img.max() - img.min() + 1e-5)
            return (img_vis * 255).astype(np.uint8)

        # Format image based on dataset type
        def format_image(img_array):
            if dataset_type == "Muds":
                return to_binary_colormap_image(img_array)
            elif dataset_type == "DynamicEarthNet":
                return to_class_colormap_image(img_array)
            return None

        # --- Log RGB input ---
        if input_rgb is not None:
            input_images = {
                f"{prefix}_input/t{t:02d}": wandb.Image(normalize_img(input_rgb[t]), caption=f"Batch {batch_idx}")
                for t in range(input_rgb.shape[0])
            }
            wandb.log(input_images)

        # --- Log IR input (only for DynamicEarthNet) ---
        if input_ir is not None:
            input_ir_images = {
                f"{prefix}_input_infrared/t{t:02d}": wandb.Image(normalize_img(np.repeat(input_ir[t], 3, axis=0)),
                                                                caption=f"Batch {batch_idx}")
                for t in range(input_ir.shape[0])
            }
            wandb.log(input_ir_images)

        # --- Log predictions ---
        if preds_np is not None:
            pred_images = {
                f"{prefix}_pred/t{t:02d}": wandb.Image(format_image(preds_np[t]), caption=f"Batch {batch_idx}")
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
                        img_fmt, caption=f"Batch {batch_idx}"
                    )
            if gt_images:
                wandb.log(gt_images)

        # --- Log temporal stats ---
        if preds_np is not None:
            temporal_images = {
                f"{prefix}_temporal/mean": wandb.Image(format_image(np.mean(preds_np, axis=0)), caption=f"Batch {batch_idx}"),
                f"{prefix}_temporal/std": wandb.Image(format_image(np.std(preds_np, axis=0)), caption=f"Batch {batch_idx}"),
                f"{prefix}_temporal/change": wandb.Image(format_image(preds_np[-1] - preds_np[0]), caption=f"Batch {batch_idx}")
            }
            wandb.log(temporal_images)


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
    """Return a matplotlib figure of a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    plt.tight_layout()
    return fig