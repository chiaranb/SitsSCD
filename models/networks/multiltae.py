"""
Multi-temporal U-TAE Implementation
Inspired by U-TAE Implementation (Vivien Sainte Fare Garnot (github/VSainteuf))
"""
import numpy as np
import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoder


class MultiLTAE(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        dropout=0.2,
        T=730,
        offset=0,
        return_att=False,
        positional_encoding=True
    ):
        """
        Lightweight Temporal Attention Encoder (L-TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared L-TAE is applied to all pixel positions of the image sequence.
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            dropout (float): dropout
            T (int): Period to use for the positional encoding.
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(MultiLTAE, self).__init__()
        self.in_channels = in_channels
        self.mlp = [in_channels, in_channels]
        self.return_att = return_att
        self.n_head = n_head
        self.d_model = in_channels
        assert self.mlp[0] == self.d_model
        # Positional encoding
        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head, offset=offset
            )
        else:
            self.positional_encoder = None
        # Attention heads
        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )
        # Normalization layers
        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.mlp[-1],
        )
        # MLP
        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1) # Normalize input

        if self.positional_encoder is not None: # Add positional encoding if enabled
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        # Apply multi-head attention
        out, attn = self.attention_heads(out, pad_mask=pad_mask)  # h x (sz_b*h*w) x t x (d//h), h x (sz_b*h*w) x t x t
        out = out.permute(1, 2, 0, 3).contiguous().view(sz_b * h * w, seq_len, -1)  # Concatenate heads
        # Apply MLP and normalization
        out = self.dropout(self.mlp(out.view(sz_b * h * w * seq_len, -1)))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, seq_len, -1).permute(0, 3, 4, 1, 2)

        attn = attn.view(self.n_head, sz_b, h, w, seq_len, seq_len).permute(
            0, 1, 4, 5, 2, 3
        )  # head x b x t x t x h x w

        if self.return_att:
            return out, attn
        else:
            return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.fc1_q = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_q.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        # Apply linear transformations to the input to get queries and keys
        q = self.fc1_q(v).view(sz_b, seq_len, n_head, d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k).permute(0, 2, 1)  # (n*b) x dk x lk

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (nxb) x lk

        # Split v into n_head parts
        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        )
        output, attn = self.attention(q, k, v, pad_mask=pad_mask) # Applies scaled dot-product attention
        attn = attn.view(n_head, sz_b, seq_len, seq_len)
        output = output.view(n_head, sz_b, seq_len, d_in // n_head)
        return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    """Input:
    - q: query, what we are trying to understand
    - k: key, what we are trying to match against
    - v: value, what we are trying to retrieve"""
    def forward(self, q, k, v, pad_mask=None):
        attn = torch.matmul(k, q) # Compute similarity between keys and queries
        attn = attn / self.temperature # Scale the attention scores

        if pad_mask is not None: # Eventually mask out padded positions
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)

        attn = self.softmax(attn) # Normalize the attention scores
        attn = self.dropout(attn) 
        output = torch.matmul(attn, v) # Compute the output as a weighted sum of values
        return output, attn
