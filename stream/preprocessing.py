import pandas as pd
import numpy as np
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import MiniBatchKMeans
import os

# ---------------- Configuration ----------------
INPUT_CSV = "/Users/chiaranguyen/Desktop/SitsSCD/stream/embeddings/embeddings_dino_sat493m.csv"
LABEL_COLUMN = "label"
EMB_PREFIX = "emb_"
META_COLUMNS = ["timestamp"]
OUTPUT_DIR = "processed_embeddings"
RANDOM_STATE = 42

PROJECTIONS = {
    "srp": SparseRandomProjection,      # Sparse Random Projection
    #"grp": GaussianRandomProjection,    # Gaussian Random Projection
}

SIZES = [1024]

# ---------------- 1. Load CSV ----------------
print("Loading CSV...")
df = pd.read_csv(INPUT_CSV)
print(f"Dataset loaded: {df.shape[0]} instances, {df.shape[1]} features")

# ---------------- 2. Remove unwanted label ----------------
#df = df[df[LABEL_COLUMN] != 6].reset_index(drop=True)
#print(f"After removing label=6: {df.shape[0]} instances remain")

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

        
        #model = proj_class(n_components=n_components, random_state=RANDOM_STATE)
        #X_proj = model.fit_transform(X_emb)

        # Normalize projected embeddings
        scaler_emb = RobustScaler()
        X_scaled = scaler_emb.fit_transform(X_emb)

        # Build final dataframe
        df_emb = pd.DataFrame(X_scaled.astype("float16"), columns=[f"emb_{i+1}" for i in range(n_components)])
        df_final = pd.concat([df[other_cols].reset_index(drop=True), df_emb], axis=1)

        # Save file
        output_name = f"{OUTPUT_DIR}/emb_dino_sat493m_{proj_name}_{n_components}.csv"
        df_final.to_csv(output_name, index=False)
        print(f"Saved: {output_name} ({df_final.shape[0]} rows, {df_final.shape[1]} cols)")

print("\nAll projections completed successfully.")