import pytorch_lightning as L
import torch
import torch.nn as nn
from hydra.utils import instantiate
from pathlib import Path
import numpy as np
import wandb
import matplotlib.cm as cm
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
        # Two sets of metrics for validation and testing, one for each domain
        self.val_metrics = {'out': instantiate(cfg.val_metrics), 'in': instantiate(cfg.val_metrics)}
        self.test_metrics = {'out': instantiate(cfg.test_metrics), 'in': instantiate(cfg.test_metrics)}
        self.domain_dict = {0: 'out', 1: 'in'}

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
    def validation_step(self, batch, batch_idx, dataloader_idx):
        pred = self.model(batch)
        pred["pred"] = torch.argmax(pred["logits"], dim=2)
        loss = self.loss(pred, batch, average=True)["loss"]
        self.val_metrics[self.domain_dict[dataloader_idx]].update(pred["pred"], batch["gt"])
        self.log("val/loss", loss, sync_dist=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        for dataloader_idx in ['out', 'in']:
            metrics = self.val_metrics[dataloader_idx].compute()
            for metric_name, metric_value in metrics.items():
                self.log(
                    f"val/{metric_name}_{dataloader_idx}",
                    metric_value,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                )
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx, dataloader_idx):
        pred = self.model(batch)
        pred["pred"] = torch.argmax(pred["logits"], dim=2)

        # Save predictions to the output directory
        domain = self.domain_dict[dataloader_idx]
        output_dir = Path(self.cfg.output_dir) / "predictions" / domain
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(pred["pred"].shape[0]):
            pred_image = pred["pred"][i].cpu().numpy()
            file_path = output_dir / f"sample_{batch_idx}_{i}.npy"
            np.save(file_path, pred_image)

        self.test_metrics[domain].update(pred["pred"], batch["gt"])

        # Define logging frequency for wandb
        log_every = 40 if dataloader_idx == 0 else 10
        if batch_idx % log_every == 0:
            wandb_data = {}

            if pred["pred"].shape[0] > 0:
                sample_pred = pred["pred"][0].cpu().numpy()  # [24, H, W]
                sample_gt = batch["gt"][0].cpu().numpy() if "gt" in batch else None

                for t_idx in range(sample_pred.shape[0]):
                    pred_t = sample_pred[t_idx]
                    wandb_data[f"pred/{domain}/t{t_idx:02d}"] = wandb.Image(
                        to_binary_colormap_image(pred_t),
                        caption=f"pred/{domain}/T{t_idx}/(Batch {batch_idx})",
                        mode="L"
                    )

                    if sample_gt is not None and t_idx < sample_gt.shape[0]:
                        gt_t = sample_gt[t_idx]
                        wandb_data[f"gt/{domain}/t{t_idx:02d}"] = wandb.Image(
                            to_binary_colormap_image(gt_t),
                            caption=f"gt/{domain}/T{t_idx}/(Batch {batch_idx})",
                            mode="L"
                        )

                # Temporal stats
                temporal_mean = np.mean(sample_pred, axis=0)
                temporal_std = np.std(sample_pred, axis=0)
                change = sample_pred[-1] - sample_pred[0]

                wandb_data[f"temporal/{domain}/mean"] = wandb.Image(
                    to_binary_colormap_image(temporal_mean),
                    caption=f"{domain} Temporal Mean (Batch {batch_idx})",
                    mode="L",
                )
                wandb_data[f"temporal/{domain}/std"] = wandb.Image(
                    to_binary_colormap_image(temporal_std),
                    caption=f"{domain} Temporal Std (Batch {batch_idx})",
                    mode="L",
                )
                wandb_data[f"temporal/{domain}/change"] = wandb.Image(
                    to_binary_colormap_image(change),
                    caption=f"{domain} Change (T23 - T00) (Batch {batch_idx})",
                    mode="L"
                )

            wandb.log(wandb_data)
            
        
    def on_test_epoch_end(self):
        for dataloader_idx in [0, 1]:
            if dataloader_idx in self.domain_dict:
                domain_name = self.domain_dict[dataloader_idx]
                metrics = self.test_metrics[domain_name].compute()
                
                print(f"\n=== DATALOADER {dataloader_idx} ({domain_name.upper()}) ===")
                for metric_name, metric_value in metrics.items():
                    print(f"{metric_name}: {metric_value}")
                    self.log(
                        f"test/{metric_name}_{dataloader_idx}",
                        metric_value,
                        sync_dist=True,
                        on_step=False,
                        on_epoch=True,
                    )

    def configure_optimizers(self):
        if self.cfg.optimizer.exclude_ln_and_biases_from_weight_decay:
            parameters_names_wd = get_parameter_names(self.model, [nn.LayerNorm])
            parameters_names_wd = [
                name for name in parameters_names_wd if "bias" not in name
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n in parameters_names_wd
                    ],
                    "weight_decay": self.cfg.optimizer.optim.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n not in parameters_names_wd
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = instantiate(
                self.cfg.optimizer.optim, optimizer_grouped_parameters
            )
        else:
            optimizer = instantiate(self.cfg.optimizer.optim, self.model.parameters())
        scheduler = instantiate(self.cfg.lr_scheduler)(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.global_step)


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    Taken from HuggingFace transformers.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

# Function to convert a 2D array to a binary colormap image
def to_binary_colormap_image(array):
    """Converte una mappa 2D in un'immagine RGB con colormap 'binary'."""
    fig, ax = plt.subplots(figsize=(1.28, 1.28), dpi=100)  # Output 128x128
    ax.axis("off")
    ax.imshow(array, cmap="binary")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    buf.seek(0)
    return Image.open(buf)
