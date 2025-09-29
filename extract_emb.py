"""
Estrazione embeddings temporali da MultiUTAE
e salvataggio in CSV con timestamp per approccio streaming / prequential.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from models.networks.multiutae import MultiUTAE  


class MultiUTAETemporalExtractor(nn.Module):
    """
    Estrae embeddings da MultiUTAE mantenendo la dimensione temporale (B x T x C).
    """
    def __init__(self, utae: nn.Module, pool='avg'):
        super().__init__()
        self.ut = utae
        assert pool in ('avg', 'max')
        self.pool = pool

    def forward(self, batch):
        x = batch["data"].float()      # B x T x C x H x W
        batch_positions = batch["positions"]
        idx = batch["idx"]
        B, T, C, H, W = x.shape
        print(f"Input shape: {x.shape}")

        # Pad mask
        pad_mask = ((x == self.ut.pad_value).all(dim=-1).all(dim=-1).all(dim=-1))  # BxT
        out = self.ut.in_conv.smart_forward(x)
        feature_maps = [out]
        
        # Spatial encoder
        for i in range(self.ut.n_stages - 1):
            out = self.ut.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)
        feat_last = feature_maps[-1]  # B x T x Cf x Hf x Wf

        # Temporal encoder
        out_temporal, att = self.ut.temporal_encoder(
            feat_last, batch_positions=batch_positions, pad_mask=pad_mask
        )

        # Pooling spaziale per timestep
        if self.pool == 'avg':
            emb = out_temporal.mean(dim=[-2, -1])  # B x T x Cf
        else:
            emb = out_temporal.amax(dim=[-2, -1])  # B x T x Cf

        return {"embeddings": emb, "pad_mask": pad_mask, "att": att, "idx": idx, "positions": batch_positions}


def save_temporal_embeddings(batch_meta, embeddings, csv_path, mode='a'):
    """
    Salva embeddings temporali in CSV.
    batch_meta: lista di dict per ogni sample (lunghezza B)
    embeddings: tensor B x T x C
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    B, T, C = embeddings.shape
    print(f"Salvataggio embeddings: {B} samples, {T} timesteps, {C} features ciascuno.")

    rows = []
    for i in range(B):
        meta = batch_meta[i]
        for t in range(T):
            row = {
                "idx": int(meta["idx"]),
                "timestamp": int(meta["positions"][t]),
            }
            for k in range(C):
                row[f"emb_{k}"] = float(embeddings[i, t, k])
            rows.append(row)

    df = pd.DataFrame(rows)
    header = True if mode == 'w' else False
    df.to_csv(csv_path, index=False, mode=mode, header=header)


def extract_embeddings_from_dataloader(dataloader, utae_model, csv_path, pool='avg'):
    """
    Itera sul DataLoader, estrae embeddings e salva in CSV.
    """
    extractor = MultiUTAETemporalExtractor(utae_model, pool=pool)
    extractor.eval()
    mode = 'w'

    with torch.no_grad():
        for batch in dataloader:
            out = extractor(batch)
            emb = out['embeddings']  # B x T x F
            idxs = out["idx"].detach().cpu().numpy()  # B
            positions = out["positions"].detach().cpu().numpy()  # B x T

            # Prepara i metadati per ogni sample
            batch_meta = [
                {"idx": int(idxs[i]), "positions": positions[i]} for i in range(len(idxs))
            ]

            save_temporal_embeddings(batch_meta, emb, csv_path, mode=mode)
            mode = 'a'  # dopo il primo batch, append


if __name__ == "__main__":
    import argparse
    from data.data import DynamicEarthNet

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="temporal_embeddings.csv", help="Path CSV output")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pool", type=str, choices=['avg', 'max'], default='avg')
    args = parser.parse_args()

    # Istanzia dataset e DataLoader
    dataset = DynamicEarthNet(
        path="/teamspace/studios/this_studio/SitsSCD/datasets/DynamicEarthNet_temporal",
        split='train',
        domain_shift_type='temporal',
        train_length=12,
        date_aug_range=0
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    utae_model = MultiUTAE(
        input_dim=4,          
        num_classes=6,         
        in_features=512
    )
    utae_model.eval()

    extract_embeddings_from_dataloader(dataloader, utae_model, args.csv_path, pool=args.pool)
    print(f"Embeddings temporali salvati su {args.csv_path}")