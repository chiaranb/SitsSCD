"""
Estrazione embeddings temporali da MultiUTAE
e salvataggio in CSV con timestamp per approccio streaming / prequential.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.networks.multiutae import MultiUTAE  

class MultiUTAETemporalExtractor(nn.Module):
    """
    Estrae embeddings temporali da MultiUTAE, mantenendo la dimensione temporale (B x T x C),
    e aggiunge le label calcolate come classe più frequente nella patch.
    """
    def __init__(self, utae: nn.Module, pool='avg'):
        super().__init__()
        self.ut = utae
        assert pool in ('avg', 'max')
        self.pool = pool

    def forward(self, batch):
        x = batch["data"].float()       # [B, T, C, H, W]
        batch_positions = batch["positions"].long()  # timestamps
        sits_id = batch["sits_id"]
        idx = batch["idx"]
        gt = batch["gt"]                # [B, T, H, W]

        # --- Mask di padding ---
        pad_mask = ((x == self.ut.pad_value).all(dim=-1).all(dim=-1).all(dim=-1))  # [B, T]

        # --- Forward encoder spaziale ---
        out = self.ut.in_conv.smart_forward(x)
        feature_maps = [out]
        for i in range(self.ut.n_stages - 1):
            out = self.ut.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)
        feat_last = feature_maps[-1]  # [B, T, Cf, Hf, Wf]

        # --- Encoder temporale ---
        out_temporal, att = self.ut.temporal_encoder(
            feat_last, batch_positions=batch_positions, pad_mask=pad_mask
        )

        # --- Pooling spaziale per timestep ---
        if self.pool == 'avg':
            emb = out_temporal.mean(dim=[-2, -1])  # [B, T, Cf]
        else:
            emb = out_temporal.amax(dim=[-2, -1])  # [B, T, Cf]

        # --- Calcolo label per timestep: moda della GT (classe più frequente nella patch) ---
        labels = self.compute_patch_majority_label(gt, num_classes=6)  # [B, T]

        return {
            "embeddings": emb,               # [B, T, C]
            "labels": labels,                # [B, T]
            "sits_id": sits_id,
            "idx": idx,
            "positions": batch_positions,    # [B, T]
            "gt": gt                         # [B, T, H, W]
        }

    @staticmethod
    def compute_patch_majority_label(gt, num_classes):
        """
        Calcola per ogni patch (B,T,H,W) la classe più frequente.
        Restituisce tensor [B,T].
        """
        B, T, H, W = gt.shape
        gt_flat = gt.view(B, T, H * W)
        labels = torch.zeros((B, T), dtype=torch.long, device=gt.device)
        for b in range(B):
            for t in range(T):
                bincount = torch.bincount(gt_flat[b, t], minlength=num_classes)
                labels[b, t] = bincount.argmax()
        return labels

def save_temporal_embeddings(batch_meta, embeddings, csv_path, mode='a'):
    """
    Salva embeddings temporali in CSV.
    batch_meta: lista di dict per ogni sample (lunghezza B)
    embeddings: tensor B x T x C
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    B, T, C = embeddings.shape
    rows = []
    for i in range(B):
        meta = batch_meta[i]
        for t in range(T):
            timestamp = int(meta["positions"][t])  # timestamp intero
            row = {
                "sits_id": meta["sits_id"],
                "idx": int(meta["idx"]),
                "timestamp": timestamp,
                "label": int(meta["label"][t]),
            }
            row.update({f"emb_{k}": float(embeddings[i, t, k]) for k in range(C)})
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, mode=mode, header=(mode == 'w'))


def extract_embeddings_from_dataloader(dataloader, utae_model, csv_path, pool='avg'):
    """
    Itera sul DataLoader, estrae embeddings e salva in CSV.
    """
    extractor = MultiUTAETemporalExtractor(utae_model, pool=pool)
    extractor.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor.to(device)
    mode = 'w'
    
    progress_bar = tqdm(dataloader, desc="Estrazione embeddings", unit="batch", ncols=100)

    with torch.no_grad():
        for batch in progress_bar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = extractor(batch)

            emb = out['embeddings']  # [B, T, C]
            labels = out['labels'].cpu().numpy()  # [B, T]
            sits_id = out['sits_id'].cpu().numpy()  # [B]
            idxs = out['idx'].cpu().numpy()  # [B]
            positions = out['positions'].cpu().numpy()  # [B, T]

            # Prepara i metadati per ogni sample
            batch_meta = [
                {"sits_id": int(sits_id[i]), "idx": int(idxs[i]), "positions": positions[i], "label": labels[i]} 
                for i in range(len(idxs))
            ]

            save_temporal_embeddings(batch_meta, emb, csv_path, mode=mode)
            mode = 'a'  # dopo il primo batch, append

            progress_bar.set_postfix({
                "batch_size": emb.shape[0],
                "timesteps": emb.shape[1],
                "features": emb.shape[2]
            })
    progress_bar.close()

if __name__ == "__main__":
    import argparse
    from data.data import DynamicEarthNet

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="embeddings_test.csv", help="Path CSV output")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pool", type=str, choices=['avg', 'max'], default='avg')
    args = parser.parse_args()

    # Istanzia dataset e DataLoader
    dataset = DynamicEarthNet(
        path="/teamspace/studios/this_studio/SitsSCD/datasets/DynamicEarthNet_Test",
        split='train',
        domain_shift_type='temporal',
        train_length=12,
        img_size=64,
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
    # Count total embeddings
    df = pd.read_csv(args.csv_path)
    print(f"Totale embeddings estratti: {len(df)}")
    print(f"Embeddings temporali salvati su {args.csv_path}")