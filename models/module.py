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

class SitsScdModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = instantiate(cfg.network.instance)
        self.loss = instantiate(cfg.loss.instance)
        self.ignore_index = self.loss.ignore_index
        self.val_metrics = instantiate(cfg.val_metrics)
        self.test_metrics = instantiate(cfg.test_metrics)
        self.dataset = self.cfg.dataset

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
        self.val_metrics.reset()
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pred = self.model(batch)
        pred["pred"] = torch.argmax(pred["logits"], dim=2)

        self.save_predictions(pred["pred"], batch_idx)
        self.test_metrics.update(pred["pred"], batch["gt"])
        self.log_wandb_images(pred["pred"], batch["gt"], batch_idx, batch["data"], prefix="test", dataset_type=self.dataset)
    
    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        print("\n=== TEST EPOCH END ===")
        self.log_metrics(metrics, prefix="test")
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

    def log_wandb_images(self, preds, gt, batch_idx, data, prefix="test", dataset_type="muds"):
        if prefix in ("test", "val"):
            preds_np = preds[0].cpu().numpy()
        gt_np = gt[0].cpu().numpy() if gt is not None else None
        input_np = data[0].cpu().numpy() if data is not None else None

        if input_np is not None:
            input_images = {}
            for t in range(input_np.shape[0]):
                img = np.moveaxis(input_np[t], 0, -1)  # C,H,W → H,W,C
                img_vis = (img - img.min()) / (img.max() - img.min() + 1e-5)
                img_vis = (img_vis * 255).astype(np.uint8)
                input_images[f"{prefix}_input/t{t:02d}"] = wandb.Image(
                    img_vis, caption=f"Batch {batch_idx}"
                )
            wandb.log(input_images)

        # Helper function to format images based on dataset type
        def format_image(img_array):
            if dataset_type == "muds":
                return to_binary_colormap_image(img_array)
            elif dataset_type == "dynamicearthnet":
                img_vis = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-5)
                return (img_vis * 255).astype(np.uint8)


        # Log pred
        if prefix in ("test", "val"):
            pred_images = {}
            for t in range(preds_np.shape[0]):
                pred_images[f"{prefix}_pred/t{t:02d}"] = wandb.Image(
                    format_image(preds_np[t]), caption=f"Batch {batch_idx}"
                )
            wandb.log(pred_images)

        # Log gt
        if gt_np is not None:
            gt_images = {}
            for t in range(gt_np.shape[0]):
                gt_images[f"{prefix}_gt/t{t:02d}"] = wandb.Image(
                    format_image(gt_np[t]), caption=f"Batch {batch_idx}"
                )
            wandb.log(gt_images)

        # Log temporal stats 
        if prefix in ("test", "val"):
            temporal_images = {
                f"{prefix}_temporal/mean": wandb.Image(
                    format_image(np.mean(preds_np, axis=0)), caption=f"Batch {batch_idx}"
                ),
                f"{prefix}_temporal/std": wandb.Image(
                    format_image(np.std(preds_np, axis=0)), caption=f"Batch {batch_idx}"
                ),
                f"{prefix}_temporal/change": wandb.Image(
                    format_image(np.max(preds_np, axis=0) - np.min(preds_np, axis=0)), caption=f"Batch {batch_idx}"
                )
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