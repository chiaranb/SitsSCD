import numpy as np
import pandas as pd
from os.path import join
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------
EMBEDDINGS_CSV = "temporal_embeddings_test.csv"  # file embeddings
LABELS_FOLDER = "labels_embeddings"                # cartella con sits_id_idx.npy
OUTPUT_CSV = "pixelwise_embeddings_test_2.csv"           # file CSV finale
SUBSAMPLE = 32                                     # 1 = tutti i pixel, >1 = subsampling

# ----------------------------
# LOAD EMBEDDINGS CSV
# ----------------------------
df = pd.read_csv(EMBEDDINGS_CSV)
cols_emb = [c for c in df.columns if c.startswith("emb_")]

rows_out = []

# ----------------------------
# ITERA SU OGNI RIGA
# ----------------------------
for idx_row, row in tqdm(df.iterrows(), total=len(df)):
    sits_id = int(row["sits_id"])
    idx_patch = int(row["idx"])
    timestamp = int(row["timestamp"])
    timestamp = timestamp  # torna a 0-11
    #print(f"Processing sits_id {sits_id}, idx {idx_patch}, timestamp {timestamp}")
    emb = row[cols_emb].values.astype(np.float32)

    # Carica label corrispondente
    label_path = join(LABELS_FOLDER, f"{sits_id}_{idx_patch}.npy")
    gt = np.load(label_path)  # shape [12, num_classes, H, W]
    label_matrix = gt[timestamp]  # [H, W]

    H, W = label_matrix.shape

    for i in range(0, H, SUBSAMPLE):
        for j in range(0, W, SUBSAMPLE):
            row_dict = {
                "sits_id": sits_id,
                "idx": idx_patch,
                "timestamp": timestamp,
                "i": i,
                "j": j
            }
            # aggiungi embeddings
            for k, val in enumerate(emb):
                row_dict[f"emb_{k}"] = val
            row_dict["label"] = int(label_matrix[i, j])
            rows_out.append(row_dict)

df_out = pd.DataFrame(rows_out)
# Reorder by timestamp
df_out = df_out.sort_values(by=["timestamp", "sits_id", "idx", "i", "j"]).reset_index(drop=True)

df_out.to_csv(OUTPUT_CSV, index=False)
print(f"Salvato CSV finale con {len(df_out)} righe su {OUTPUT_CSV}")