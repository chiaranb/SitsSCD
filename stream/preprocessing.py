import pandas as pd
import numpy as np
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
import umap
import os

# ---------------- Configuration ----------------
INPUT_CSV = "embeddings.csv"
LABEL_COLUMN = "label"
EMB_PREFIX = "emb_"
META_COLUMNS = ["timestamp"]
OUTPUT_DIR = "processed_embeddings"
RANDOM_STATE = 42

PROJECTIONS = {
    "srp": SparseRandomProjection,      # Sparse Random Projection
    "grp": GaussianRandomProjection,    # Gaussian Random Projection
}

SIZES = [256, 128, 64]

# ---------------- 1. Load CSV ----------------
print("Loading CSV...")
df = pd.read_csv(INPUT_CSV)
print(f"Dataset loaded: {df.shape[0]} instances, {df.shape[1]} features")

# ---------------- 2. Remove unwanted label ----------------
df = df[df[LABEL_COLUMN] != 6].reset_index(drop=True)
print(f"After removing label=6: {df.shape[0]} instances remain")

# ---------------- 3. Separate embeddings ----------------
embeddings_cols = [col for col in df.columns if col.startswith(EMB_PREFIX)]
other_cols = [col for col in df.columns if col not in embeddings_cols]

X_emb = df[embeddings_cols].values.astype(np.float32)
print(f"Embeddings shape: {X_emb.shape}")

# ---------------- 4. Normalize metadata ----------------
#scaler_meta = MinMaxScaler()
#df[META_COLUMNS] = scaler_meta.fit_transform(df[META_COLUMNS].values)
#print("Metadata normalized")

# ---------------- 5. Create output directory ----------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- 6. Loop over projection methods and sizes ----------------
for proj_name, proj_class in PROJECTIONS.items():
    for n_components in SIZES:
        print(f"\n=== {proj_name.upper()} projection to {n_components} dims ===")

        if proj_name == "kmeans":
            # Embedding via distances to cluster centroids
            model = proj_class(n_clusters=n_components, random_state=RANDOM_STATE, batch_size=2048)
            model.fit(X_emb)
            X_proj = model.transform(X_emb)
        elif proj_name == "umap":
            model = proj_class(n_components=n_components, random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1, metric="euclidean")
            X_proj = model.fit_transform(X_emb)
        else:
            model = proj_class(n_components=n_components, random_state=RANDOM_STATE)
            X_proj = model.fit_transform(X_emb)

        # Normalize projected embeddings
        scaler_emb = StandardScaler()
        X_scaled = scaler_emb.fit_transform(X_proj)

        # Build final dataframe
        df_emb = pd.DataFrame(X_scaled.astype("float16"), columns=[f"emb_{i+1}" for i in range(n_components)])
        df_final = pd.concat([df[other_cols].reset_index(drop=True), df_emb], axis=1)

        # Save file
        output_name = f"{OUTPUT_DIR}/emb_{proj_name}_{n_components}.csv"
        df_final.to_csv(output_name, index=False)
        print(f"Saved: {output_name} ({df_final.shape[0]} rows, {df_final.shape[1]} cols)")

print("\nAll projections completed successfully.")