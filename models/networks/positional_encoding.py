import torch
import torch.nn as nn

"""Encode time information to help the model know "when" each data point occurs."""
class PositionalEncoder(nn.Module):
    def __init__(self, d, T=730, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        # The denominator is used to scale the sinusoidal functions
        self.denom = torch.pow(
            T, 2 * torch.div(torch.arange(offset, offset + d).float(), 2, rounding_mode='floor') / (d+offset)
        )
        self.updated_location = False

    def forward(self, batch_positions):
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True
        # Compute the sinusoidal encoding for the batch positions
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        # Apply sine and cosine functions to even and odd dimensions
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table
