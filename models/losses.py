"""
Patch-level Focal Loss
 - logits: [B, T, C]
 - gt: [B, T]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        self.size_average = size_average

    def forward(self, x, y):
        """
        Args:
            x: dict that contains "logits": torch.Tensor [B, T, C]
            y: dict that contains "gt": torch.Tensor [B, T]
        Returns:
            torch.Tensor: scalar focal loss
        """
        prediction = x["logits"]  # [B, T, C]
        target = y["gt"]          # [B, T]

        if prediction.ndim == 5:
            B, T, C, H, W = prediction.shape
        elif prediction.ndim == 3:
            B, T, C = prediction.shape
            H = W = 1  # placeholder per compatibilità
            prediction = prediction.view(B, T, C, H, W)
        else:
            raise ValueError(f"Unexpected prediction shape: {prediction.shape}")
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous().view(B * T * H * W, C)
        target = target.contiguous().view(B * T * H * W)

        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            prediction = prediction[valid_mask]
            target = target[valid_mask]

        target = target.unsqueeze(1)
        logpt = F.log_softmax(prediction, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != prediction.data.type():
                self.alpha = self.alpha.type_as(prediction.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        return loss.mean() if self.size_average else loss.sum()


LOSSES = {
    "focal": FocalLoss,
}

AVERAGE = {False: lambda x: x, True: lambda x: x.mean(dim=-1)}


class Losses(nn.Module):
    """Meta-loss container for multiple weighted losses (patch-level)."""

    def __init__(self, mix={}, ignore_index=None):
        super(Losses, self).__init__()
        assert len(mix)
        self.ignore_index = ignore_index
        self.init_losses(mix)

    def init_losses(self, mix):
        self.loss = {}
        for name, weight in mix.items():
            name = name.lower()
            if name not in LOSSES:
                raise KeyError(f"Loss {name} not found. Available: {LOSSES.keys()}")
            self.loss[name] = (LOSSES[name](ignore_index=self.ignore_index), weight)

    def forward(self, x, y, average=True):
        """
        Args:
            x: dict that contains "logits": torch.Tensor [B, T, C]
            y: dict that contains "gt": torch.Tensor [B, T]
        Returns:
            dict of individual and total loss values
        """
        losses = {n: AVERAGE[average](f(x, y)) for n, (f, _) in self.loss.items()}
        losses["loss"] = sum(losses[n] * w for n, (_, w) in self.loss.items())
        return losses