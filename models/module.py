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
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pred = self.model(batch)
        pred["pred"] = torch.argmax(pred["logits"], dim=2)
        loss = self.loss(pred, batch, average=True)["loss"]
        self.val_metrics.update(pred["pred"], batch["gt"])
        self.log("val/loss", loss, sync_dist=True, on_step=False, on_epoch=True)
    
    def on_validation_epoch_end(self):
        computed = self.val_metrics.compute()
        self.log_metrics(computed, prefix="val", suffix="all")
        self.val_metrics.reset()
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pred = self.model(batch)
        pred["pred"] = torch.argmax(pred["logits"], dim=2)

        self.save_predictions(pred["pred"], batch_idx, "all")
        self.test_metrics["all"].update(pred["pred"], batch["gt"])

        if batch_idx % 10 == 0:
            self.log_wandb_images(pred["pred"], batch["gt"], batch_idx, None)
    
    def on_test_epoch_end(self):
        metrics = self.test_metrics["all"].compute()
        print("\n=== TEST EPOCH END ===")
        self.log_metrics(metrics, prefix="test", suffix="all")
        self.test_metrics["all"].reset()

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

    def log_metrics(self, metrics, prefix, suffix):
        for name, value in metrics.items():
            self.log(f"{prefix}/{name}_{suffix}", value, sync_dist=True, on_step=False, on_epoch=True)

    def log_wandb_images(self, preds, gt, batch_idx, domain=None):
        preds_np = preds[0].cpu().numpy()
        gt_np = gt[0].cpu().numpy() if gt is not None else None
        wandb_data = {}

        for t in range(preds_np.shape[0]):
            wandb_data[f"pred/{domain}/t{t:02d}"] = wandb.Image(
                to_binary_colormap_image(preds_np[t]),
                caption=f"pred/{domain}/T{t}/(Batch {batch_idx})"
            )
            if gt_np is not None and t < gt_np.shape[0]:
                wandb_data[f"gt/{domain}/t{t:02d}"] = wandb.Image(
                    to_binary_colormap_image(gt_np[t]),
                    caption=f"gt/{domain}/T{t}/(Batch {batch_idx})"
                )

        wandb_data[f"temporal/{domain}/mean"] = wandb.Image(
            to_binary_colormap_image(np.mean(preds_np, axis=0)),
            caption=f"{domain} Temporal Mean (Batch {batch_idx})"
        )
        wandb_data[f"temporal/{domain}/std"] = wandb.Image(
            to_binary_colormap_image(np.std(preds_np, axis=0)),
            caption=f"{domain} Temporal Std (Batch {batch_idx})"
        )
        wandb_data[f"temporal/{domain}/change"] = wandb.Image(
            to_binary_colormap_image(preds_np[-1] - preds_np[0]),
            caption=f"{domain} Change (T23 - T00) (Batch {batch_idx})"
        )
        wandb.log(wandb_data)


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


def to_binary_colormap_image(array):
    fig, ax = plt.subplots(figsize=(1.28, 1.28), dpi=100)
    ax.axis("off")
    ax.imshow(array, cmap="binary")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)