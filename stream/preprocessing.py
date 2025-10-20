import pandas as pd
from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# ---------------- Configuration ----------------
INPUT_CSV = "embeddings_sorted.csv"          
OUTPUT_CSV = "embeddings_test.csv"  
N_COMPONENTS = 128                        
RANDOM_STATE = 42                         
LABEL_COLUMN = "label"                   
EMB_PREFIX = "emb_"                      # prefix for embedding columns
META_COLUMNS = ["timestamp"]   # metadata columns to normalize

# ---------------- 1. Load CSV ----------------
print("Loading CSV...")
df = pd.read_csv(INPUT_CSV)
print(f"Dataset loaded: {df.shape[0]} instances, {df.shape[1]} features")

# ---------------- 2. Remove rows with label == 6 ----------------
df = df[df[LABEL_COLUMN] != 6].reset_index(drop=True)
print(f"After removing label=6: {df.shape[0]} instances remain")

# ---------------- 3. Separate features ----------------
embeddings_cols = [col for col in df.columns if col.startswith(EMB_PREFIX)]
other_cols = [col for col in df.columns if col not in embeddings_cols]

X_emb = df[embeddings_cols].values
y = df[LABEL_COLUMN].values
print(f"Embeddings shape: {X_emb.shape}")

# ---------------- 4. Dimensionality reduction for embeddings ----------------
print(f"Sparse random projection to {N_COMPONENTS} dimensions...")
srp = SparseRandomProjection(n_components=N_COMPONENTS, density='auto', random_state=RANDOM_STATE)
X_proj = srp.fit_transform(X_emb)
print(f"Reduction completed: {X_emb.shape[1]} → {X_proj.shape[1]} dimensions")

# ---------------- 5. Normalize embeddings ----------------
print("Normalizing embeddings...")
scaler_emb = StandardScaler()
X_scaled = scaler_emb.fit_transform(X_proj)
print("Embeddings normalized")

# ---------------- 6. Normalize metadata columns (timestamp, sits_id) ----------------
print("Normalizing metadata columns (timestamp, sits_id)...")
scaler_meta = MinMaxScaler()
df[META_COLUMNS] = scaler_meta.fit_transform(df[META_COLUMNS].values)
print("Metadata columns normalized")

# ---------------- 7. Reconstruct final dataframe ----------------
df_embeddings = pd.DataFrame(X_scaled.astype("float16"), columns=[f"emb_{i+1}" for i in range(N_COMPONENTS)])
df_final = pd.concat([df[other_cols].reset_index(drop=True), df_embeddings], axis=1)

# ---------------- 8. Save preprocessed file ----------------
df_final.to_csv(OUTPUT_CSV, index=False)
print(f"Preprocessed file saved to '{OUTPUT_CSV}' ({df_final.shape[0]} rows, {df_final.shape[1]} features)")