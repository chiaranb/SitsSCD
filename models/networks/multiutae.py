"""
Multi-temporal U-TAE Implementation
Inspired by U-TAE Implementation (Vivien Sainte Fare Garnot (github/VSainteuf))
"""
import torch
import torch.nn as nn

from .multiltae import MultiLTAE
from .blocks import ConvBlock, DownConvBlock, UpConvBlock


class MultiUTAE(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        in_features,
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        agg_mode="att_group",
        encoder_norm="group",
        n_head=16,
        d_k=4,
        pad_value=0,
        padding_mode="reflect",
        T=730,
        offset=0
    ):
        """
        Multi-temporal U-TAE architecture for multi-stamp spatio-temporal encoding of satellite image time series.
        Args:
            input_dim (int): Number of channels in the input images.
            num_classes (int): Number of classes i.e. number of output channels.
            in_features (int): Feature size at the innermost stage.
            str_conv_k (int): Kernel size of the strided up and down convolutions.
            str_conv_s (int): Stride of the strided up and down convolutions.
            str_conv_p (int): Padding of the strided up and down convolutions.
            agg_mode (str): Aggregation mode for the skip connections. Can either be:
                - att_group (default) : Attention weighted temporal average, using the same
                channel grouping strategy as in the LTAE. The attention masks are bilinearly
                resampled to the resolution of the skipped feature maps.
                - att_mean : Attention weighted temporal average,
                 using the average attention scores across heads for each date.
                - mean : Temporal average excluding padded dates.
            encoder_norm (str): Type of normalisation layer to use in the encoding branch. Can either be:
                - group : GroupNorm (default)
                - batch : BatchNorm
                - instance : InstanceNorm
            n_head (int): Number of heads in LTAE.
            d_k (int): Key-Query space dimension
            pad_value (float): Value used by the dataloader for temporal padding.
            padding_mode (str): Spatial padding strategy for convolutional layers (passed to nn.Conv2d).
            T (int): Period to use for the positional encoding.
            offset (int): Offset to use for the positional encoding. 
        """
        super().__init__()
        self.encoder_widths = [in_features // 2, in_features // 2, in_features // 2, in_features]
        self.decoder_widths = [in_features // 4, in_features // 4, in_features // 2, in_features]
        self.n_stages = len(self.encoder_widths)
        self.enc_dim = (
            self.decoder_widths[0] if self.decoder_widths is not None else self.encoder_widths[0]
        )
        self.stack_dim = (
            sum(self.decoder_widths) if self.decoder_widths is not None else sum(self.encoder_widths)
        )
        self.pad_value = pad_value

        if self.decoder_widths is not None:
            assert len(self.encoder_widths) == len(self.decoder_widths)
            assert self.encoder_widths[-1] == self.decoder_widths[-1]
        else:
            self.decoder_widths = self.encoder_widths

        in_conv_kernels = [input_dim] + [self.encoder_widths[0], self.encoder_widths[0]]
        # First convolution block to encode input
        self.in_conv = ConvBlock(
            nkernels=in_conv_kernels,
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        # Spatial encoder (downsampling path)
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=self.encoder_widths[i],
                d_out=self.encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode
            )
            for i in range(self.n_stages - 1)
        )
        # Spatial decoder (upsampling path)
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=self.decoder_widths[i],
                d_out=self.decoder_widths[i - 1],
                d_skip=self.encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm="group",
                padding_mode=padding_mode
            )
            for i in range(self.n_stages - 1, 0, -1)
        )
        # Applies multi-head temporal attention
        self.temporal_encoder = MultiLTAE(
            in_channels=self.encoder_widths[-1],
            n_head=n_head,
            return_att=True,
            d_k=d_k,
            T=T,
            offset=offset
        )
        # Aggregates attention-weighted skip features
        self.temporal_aggregator = Temporal_Aggregator(mode=agg_mode)
        # Final convolution block to produce output logits
        self.out_conv = ConvBlock(nkernels=[self.decoder_widths[0]] + [in_features // 4, num_classes],
                                  padding_mode=padding_mode,
                                  norm='None')

    def forward(self, batch):
        x = batch["data"].float()  
        batch_positions = batch["positions"]
        batch_size, seq_len, c, h, w = x.size() # x is BxTxCxHxW
        if batch_positions is None:
            batch_positions = torch.tensor(range(1, x.shape[1]+1),
                                           dtype=torch.long,
                                           device=x.device)[None].expand(x.shape[0], -1, -1, -1, -1)
        # Pad mask
        pad_mask = (
            (x == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask
        out = self.in_conv.smart_forward(x) # Apply initial convolution block
        feature_maps = [out]

        # SPATIAL ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)

        # TEMPORAL ENCODER
        out, att = self.temporal_encoder(
            feature_maps[-1], batch_positions=batch_positions, pad_mask=pad_mask
        )

        # SPATIAL DECODER
        for i in range(self.n_stages - 1):
            # Aggregate temporal features across time steps
            skip = self.temporal_aggregator( 
                feature_maps[-(i + 2)], pad_mask=pad_mask, attn_mask=att
            )
            out = self.up_blocks[i](out, skip) # Apply upsampling block
        # Final convolution block to produce output logits
        out = self.out_conv(out.view(batch_size * seq_len, -1, h, w)).view(batch_size, seq_len, -1, h, w)
        return {"logits": out}

"""Used to aggregate temporal features across time steps. It uses different strategies based on the mode.
- 'att_group': Uses grouped multi-head attention to weight each time step.
- 'att_mean': Averages the attention masks across heads and uses them to weight features.
- 'mean': Averages the features across time steps, excluding padded dates."""
class Temporal_Aggregator(nn.Module):
    def __init__(self, mode="mean"):
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)

                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out

            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out

            elif self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out

        else:  # No pad mask
            if self.mode == "att_group":
                n_heads, b, t, _, h, w = attn_mask.shape
                attn = attn_mask  # [n_heads, B, T, T, H, W]

                out_chunks = []
                x_chunks = x.chunk(n_heads, dim=2)  # -> list of [B, T, C/h, H, W]

                for h in range(n_heads):
                    x_h = x_chunks[h]                      # [B, T, C/h, H, W]
                    attn_h = attn[h]                       # [B, T, T, H, W]

                    weighted = []
                    for t in range(attn_h.shape[1]):
                        attn_weights = attn_h[:, t]        # [B, T, H, W]
                        attn_weights = attn_weights.unsqueeze(2)  # [B, T, 1, H, W]

                        B, T, _, H, W = attn_weights.shape
                        attn_weights = attn_weights.reshape(B * T, 1, H, W)
                        attn_weights = torch.nn.functional.interpolate(
                            attn_weights,
                            size=x_h.shape[-2:],  # (H_out, W_out)
                            mode='bilinear',
                            align_corners=False
                        )
                        attn_weights = attn_weights.view(B, T, 1, x_h.shape[-2], x_h.shape[-1])

                        attn_weights = attn_weights.expand(-1, -1, x_h.shape[2], -1, -1)  # [B, T, C/h, H, W]
                        weighted_t = (attn_weights * x_h).sum(dim=1)  # [B, C/h, H, W]
                        weighted.append(weighted_t)

                    out_chunks.append(torch.stack(weighted, dim=1))  # [B, T, C/h, H, W]

                out = torch.cat(out_chunks, dim=2)  # -> [B, T, C, H, W]
                return out

            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out

            elif self.mode == "mean":
                return x.mean(dim=1)


def create_mlp(in_ch, out_ch, n_hidden_units, n_layers):
    if n_layers > 0:
        seq = [nn.Linear(in_ch, n_hidden_units), nn.ReLU(True)]
        for _ in range(n_layers - 1):
            seq += [nn.Linear(n_hidden_units, n_hidden_units), nn.ReLU(True)]
        seq += [nn.Linear(n_hidden_units, out_ch)]
    else:
        seq = [nn.Linear(in_ch, out_ch)]
    return nn.Sequential(*seq)
