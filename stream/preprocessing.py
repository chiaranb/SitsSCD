import pandas as pd
from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import StandardScaler
import joblib

# Configuration
INPUT_CSV = "embeddings_sorted.csv"          
OUTPUT_CSV = "embeddings_preprocessed.csv"  
N_COMPONENTS = 128                        # target dimensionality
RANDOM_STATE = 42                         # for reproducibility
LABEL_COLUMN = "label"                   # name of the label column
EMB_PREFIX = "emb_"                     # prefix of embedding columns

# 1. Load data
print("Loading CSV...")
df = pd.read_csv(INPUT_CSV)
print(f"Dataset loaded: {df.shape[0]} instances, {df.shape[1]} features")

# Separate features and labels
embeddings_cols = [col for col in df.columns if col.startswith(EMB_PREFIX)]
other_cols = [col for col in df.columns if col not in embeddings_cols]

X = df[embeddings_cols].values
y = df[LABEL_COLUMN].values
print(f"Features shape: {X.shape}")

# 2. Sparse Random Projection
print(f"Sparse random projection to {N_COMPONENTS} dimensions...")
srp = SparseRandomProjection(n_components=N_COMPONENTS, density='auto', random_state=RANDOM_STATE)
X_proj = srp.fit_transform(X)
print(f"Reduction completed: {X.shape[1]} → {X_proj.shape[1]} dimensions")

# 3. Normalization (StandardScaler)
print("Normalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_proj)
print("Normalization completed")

df_embeddings = pd.DataFrame(X_scaled.astype("float16"), columns=[f"emb_{i+1}" for i in range(N_COMPONENTS)])
df_final = pd.concat([df[other_cols].reset_index(drop=True), df_embeddings], axis=1)

# 4. Save preprocessed data
print("Saving preprocessed file...")
df_final.to_csv(OUTPUT_CSV, index=False)
print(f"File saved to '{OUTPUT_CSV}' ({df_final.shape[0]} rows, {df_final.shape[1]} features)")
