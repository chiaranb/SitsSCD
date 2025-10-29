"""
Estrazione embeddings DENSE (per patch 16x16) da DINOv3
e salvataggio in CSV con LABEL PER PATCH e PATCH_ID UNIVOCO.
"""

import os
import sys
# Aggiungi il path se necessario
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F # Import necessario
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
from torchvision import transforms
import argparse # Importa argparse

# Assumi che DynamicEarthNet sia importabile
try:
    from data.data import DynamicEarthNet
except ImportError:
    print("ERRORE: Impossibile importare 'DynamicEarthNet' dal path 'data.data'.")
    print("Assicurati che lo script sia posizionato correttamente e che data/data.py esista.")
    # Inseriamo una classe FAKE per permettere allo script di essere analizzato
    # Ma fallirà se il vero DynamicEarthNet non viene trovato.
    class DynamicEarthNet:
        def __init__(self, *args, **kwargs):
            raise ImportError("Classe DynamicEarthNet Fittizia. Path errato.")

# ----------------------------------------------------------------------------
# 1. CLASSE EXTRACTOR (Invariata)
# ----------------------------------------------------------------------------

class DINOv3TemporalExtractor(nn.Module):
    """
    Estrae embeddings DENSE (per patch 16x16) da DINOv3
    e le relative LABEL PER PATCH 16x16.
    """
    def __init__(self, dinov3_model: nn.Module, num_classes_gt=6, input_channels_indices=[0, 1, 2], patch_size=16):
        super().__init__()
        self.model = dinov3_model
        self.model.eval()
        self.embedding_dim = self.model.num_features
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.normalize_transform = transforms.Normalize(mean=mean, std=std)
        
        self.num_classes_gt = num_classes_gt
        self.input_channels_indices = input_channels_indices
        self.patch_size = patch_size # es. 16
        
        if len(self.input_channels_indices) != 3:
            raise ValueError("DINOv3 richiede 3 canali di input.")
        
        print(f"[INFO] DINOv3Extractor (per Patch 16x16) wrapper creato. Dim. emb: {self.embedding_dim}")
        print(f"[INFO] Utilizzando i canali di input: {self.input_channels_indices}")

    def forward(self, batch):
        x = batch["data"].float()       # [B, T, C_in, H, W]
        gt = batch["gt"]                # [B, T, H, W]
        B, T, C_in, H, W = x.shape
        
        # --- Preparazione input per DINOv3 ---
        pad_mask = (x.sum(dim=[-1, -2, -3]) == 0)  # [B, T]
        x_flat = x.view(B * T, C_in, H, W)
        x_rgb = x_flat[:, self.input_channels_indices, :, :] 
        x_norm = self.normalize_transform(x_rgb)

        # --- Forward DINOv3 ---
        # Ottiene [B*T, 1(CLS) + 4(REG) + 196(PATCH), C_emb] = [B*T, 201, C_emb]
        all_tokens_flat = self.model.forward_features(x_norm) 
        
        # --- MODIFICA CHIAVE ---
        # Calcoliamo il numero atteso di patch dalla geometria dell'input
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w  # Questo sarà 196

        # Estraiamo SOLO gli ultimi 'num_patches' token.
        # Questo scarta automaticamente [CLS] e i register tokens all'inizio.
        # Slicing [-196:]
        patch_tokens_flat = all_tokens_flat[:, -num_patches:, :] # Shape: [B*T, 196, C_emb]
        # --- FINE MODIFICA ---
        
        # --- Ricostruisci dimensione temporale ---
        # 'num_patches' qui ora è 196
        emb = patch_tokens_flat.view(B, T, num_patches, self.embedding_dim) 
        emb[pad_mask] = 0.0

        # --- Calcolo label PER PATCH 16x16 ---
        # Questa funzione calcola correttamente 196 label
        labels_per_patch = self.compute_patch_majority_label(
            gt, self.patch_size, self.num_classes_gt
        ) # [B, T, 196]

        # Ora le forme combaciano: emb=[B,T,196,C] e labels=[B,T,196]
        return {
            "embeddings": emb,               # [B, T, 196, C_emb]
            "labels": labels_per_patch,      # [B, T, 196]
            "sits_id": batch["sits_id"],
            "positions": batch["positions"].long(),
        }

    @staticmethod
    def compute_patch_majority_label(gt, patch_size, num_classes):
        """
        Calcola la classe di maggioranza per OGNI patch.
        (Questa funzione è corretta, la lascio invariata)
        """
        B, T, H, W = gt.shape
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(f"La dimensione H={H}, W={W} non è divisibile per patch_size={patch_size}")

        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        num_patches_total = num_patches_h * num_patches_w
        
        gt_flat = gt.reshape(B * T, 1, H, W).float()
        patches = F.unfold(gt_flat, kernel_size=patch_size, stride=patch_size)
        labels_flat, _ = torch.mode(patches, dim=1)
        labels = labels_flat.view(B, T, num_patches_total).long()
        
        return labels

# ----------------------------------------------------------------------------
# 2. FUNZIONE DI SALVATAGGIO CSV (MODIFICATA)
# ----------------------------------------------------------------------------

def save_temporal_embeddings(batch_meta, embeddings, csv_path, mode='a', start_global_patch_id=0):
    """
    Salva embeddings temporali (per patch 16x16) in CSV con patch_id univoco.
    batch_meta: lista di dict per ogni sample (lunghezza B)
    embeddings: tensor B x T x Num_Patches x C
    start_global_patch_id: contatore globale per le patch 16x16
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    B, T, Num_Patches, C = embeddings.shape
    rows = []
    
    # Questo è il nostro contatore globale per le patch 16x16
    global_patch_id = start_global_patch_id 

    for i in range(B):
        meta = batch_meta[i] # Contiene sits_id, positions, e labels (array 2D T x 196)
        
        for t in range(T):
            timestamp = int(meta["positions"][t])
            labels_for_timestep = meta["label"][t] 
            patch_embeddings_for_timestep = embeddings[i, t]
            
            # Controlla se è un timestamp paddato
            if patch_embeddings_for_timestep.sum() == 0.0:
                continue 

            # Itera su ogni singola patch 16x16 (es. 0-195)
            for patch16_idx in range(Num_Patches):
                patch_label = int(labels_for_timestep[patch16_idx])
                
                row = {
                    "sits_id": meta["sits_id"],  # ID dell'immagine/scena/location
                    "patch_id": global_patch_id, # <-- MODIFICA: ID univoco globale
                    "timestamp": timestamp,
                    "label": patch_label,        # Label specifica della patch 16x16
                    # "patch16_index" rimosso
                }
                
                # Aggiungi le C features per questa patch 16x16
                emb_vector = patch_embeddings_for_timestep[patch16_idx]
                row.update({f"emb_{k}": float(emb_vector[k]) for k in range(C)})
                rows.append(row)
                
                global_patch_id += 1 # <-- MODIFICA: Incrementa per ogni patch 16x16 salvata

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, mode=mode, header=(mode == 'w'))
    
    return global_patch_id # <-- MODIFICA: Restituisce il nuovo contatore globale


# ----------------------------------------------------------------------------
# 3. FUNZIONE DI ESECUZIONE (MODIFICATA)
# ----------------------------------------------------------------------------

def extract_embeddings_from_dataloader(dataloader, dinov3_model, csv_path, num_classes_gt, input_channels_indices):
    """
    Itera sul DataLoader, estrae embeddings e salva in CSV.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Esecuzione su: {device}")

    dinov3_model.to(device)
    dinov3_model.eval()

    extractor = DINOv3TemporalExtractor(
        dinov3_model, 
        num_classes_gt=num_classes_gt, 
        input_channels_indices=input_channels_indices,
        patch_size=dinov3_model.patch_embed.patch_size[0] # Ottiene 16 dal modello
    )
    extractor.to(device)
    extractor.eval()

    mode = 'w'
    patch_id_counter = 0 # Contatore globale per patch 16x16
    
    progress_bar = tqdm(dataloader, desc="Estrazione embeddings DINOv3", unit="batch", ncols=100)

    with torch.no_grad():
        for batch in progress_bar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            out = extractor(batch)

            emb = out['embeddings']         # [B, T, 196, C_emb]
            labels = out['labels'].cpu().numpy() # [B, T, 196]
            sits_id = out['sits_id'].cpu().numpy()   # [B]
            positions = out['positions'].cpu().numpy() # [B, T]

            # Passiamo l'intero array di label (T, 196) per ogni sample
            batch_meta = [
                {"sits_id": int(sits_id[i]), "positions": positions[i], "label": labels[i]} 
                for i in range(len(sits_id))
            ]

            # Passiamo e riceviamo il contatore globale di patch 16x16
            patch_id_counter = save_temporal_embeddings(
                batch_meta, emb, csv_path, mode=mode, start_global_patch_id=patch_id_counter # <-- MODIFICA
            )
            mode = 'a'

            progress_bar.set_postfix({
                "batch_size": emb.shape[0],
                "timesteps": emb.shape[1],
            })
    progress_bar.close()

# ----------------------------------------------------------------------------
# 4. BLOCCO MAIN (MODIFICATO)
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Argomenti resi più robusti
    parser = argparse.ArgumentParser(description="Estrai embeddings per patch DINOv3")
    parser.add_argument("--csv_path", type=str, default="embeddings_dino_patches.csv", help="Path CSV output")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per il DataLoader")
    parser.add_argument("--data_path", type=str, default="/Users/chiaranguyen/Desktop/SitsSCD/datasets/DynamicEarthNet_DINO_Test", help="Path alla cartella DynamicEarthNet")
    parser.add_argument("--model_name", type=str, default="vit_small_patch16_dinov3", help="Nome modello 'timm' (es. vit_small_patch16_dinov3)")
    
    args = parser.parse_args()

    # --- Parametri Fissi ---
    INPUT_DIM = 4 
    NUM_CLASSES_GT = 6
    INPUT_CHANNELS_INDICES = [0, 1, 2] 
    
    # --- Dataset e dataloader ---
    print(f"Caricamento dataset da: {args.data_path}")
    try:
        dataset = DynamicEarthNet(
            path=args.data_path,
            split='train',
            domain_shift_type='temporal',
            train_length=24,
            img_size=224,    # Fondamentale per DINOv3 patch 16
            date_aug_range=0
        )
    except Exception as e:
        print(f"Errore durante il caricamento del dataset: {e}")
        print("Controlla che il path sia corretto e che 'data.data.DynamicEarthNet' sia importabile.")
        sys.exit(1)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Dimensione Dataloader: {len(dataloader)} batch")

    # --- Creazione modello DINOv3 ---
    print(f"Caricamento modello: {args.model_name}")
    try:
        dinov3_model = timm.create_model(
            args.model_name,
            pretrained=True,
            num_classes=0  # Per estrarre features
        )
        dinov3_model.eval()
    except Exception as e:
        print(f"Errore caricamento modello '{args.model_name}': {e}")
        print("Possibile causa: 'timm' non è aggiornato? Prova con: pip install --upgrade timm")
        sys.exit(1)

    # --- Esegui estrazione ---
    extract_embeddings_from_dataloader(
        dataloader, 
        dinov3_model, 
        args.csv_path,
        num_classes_gt=NUM_CLASSES_GT,
        input_channels_indices=INPUT_CHANNELS_INDICES
    )
    
    # --- Post-processing CSV (MODIFICATO) ---
    print("Ordinamento del file CSV finale...")
    try:
        df = pd.read_csv(args.csv_path)
        
        # Ordina per timestamp, poi per SITS, poi per il patch_id univoco
        df.sort_values(by=["timestamp", "sits_id", "patch_id"], inplace=True) # <-- MODIFICA
        
        df.to_csv(args.csv_path, index=False)
        
        print(f"\nOperazione completata.")
        print(f"Totale embeddings estratti (righe CSV): {len(df)}")
        print(f"Embeddings per patch 16x16 salvati su {args.csv_path}")
    except pd.errors.EmptyDataError:
        print(f"\nErrore: Il file CSV è vuoto. Controlla il dataloader e il percorso dati.")
    except Exception as e:
        print(f"Errore durante il post-processing del CSV: {e}")