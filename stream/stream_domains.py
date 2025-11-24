import wandb
from capymoa.classifier import (
    HoeffdingTree, NaiveBayes, SGDClassifier, KNN, EFDT, WeightedkNN,
    HoeffdingAdaptiveTree, LeveragingBagging, OnlineAdwinBagging, StreamingGradientBoostedTrees, AdaptiveRandomForestClassifier,
    DynamicWeightedMajority, OnlineBagging, OzaBoost, OnlineSmoothBoost, StreamingGradientBoostedTrees, StreamingRandomPatches, SAMkNN, CSMOTE
)
from capymoa.evaluation import ClassificationEvaluator
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from capymoa.instance import LabeledInstance
from capymoa.stream import Schema 
from capymoa.base import SKClassifier
import os
import json 
from sklearn import linear_model, multiclass

from metrics import StreamingChangeEvaluator, NUM_CLASSES, CLASS_NAMES
from utils import plot_confusion_matrix_image

# ---------------- Configuration ----------------
wandb.login()
PROJECT_NAME = "capymoa-spatial_temporal_domain_shift" 

ADAPT_ON_STREAM = False
PROCESSED_DIR = "/Users/chiaranguyen/Desktop/SitsSCD/stream/embeddings"
SPLITS_JSON_PATH = "/Users/chiaranguyen/Desktop/SitsSCD/stream/split.json"
SCENARIOS_TO_RUN = ["temporal_spatial"]

PATCH_ID_COLUMN_NAME = "patch_id"
LABEL_NAME = "label"
OTHER_FEATURES = ["sits_id", "timestamp"]
RANDOM_SEED = 42

# ---------------- Define models ----------------
MODELS = {
    "SGD_SVM": lambda schema: SKClassifier(
        schema=schema,
        sklearner=linear_model.SGDClassifier(random_state=RANDOM_SEED, loss="hinge")
    ),
    "Perceptron": lambda schema: SKClassifier(
        schema=schema,
        sklearner=linear_model.Perceptron(random_state=RANDOM_SEED)
    ),
    "PassiveAggressive": lambda schema: SKClassifier(
        schema=schema,
        sklearner=linear_model.PassiveAggressiveClassifier(random_state=RANDOM_SEED)
    ),
}

# ---------------- Helper: prequential loop per scenario ----------------
def run_scenario_experiment(
    df: pd.DataFrame, 
    splits: dict, 
    scenario_name: str, 
    csv_path: str
):
    print(f"--- Running scenario: {scenario_name} ---")
    
    # 1️⃣ Logica di Splitting (Spatial & Temporal)
    try:
        train_ids = splits[scenario_name]['train_ids']
        prequential_ids = splits[scenario_name]['prequential_ids']
    except KeyError:
        print(f"Errore: '{scenario_name}' non trovato nel file JSON. Salto.")
        return []

    # Initial training (2018 + 2019 @ Siti A)
    df_train = df[
        (df['sits_id'].isin(train_ids))
    ].reset_index(drop=True)

    # Prequential data (depends on the scenario)
    if scenario_name == "spatial":
        # Prequential: 2018 + 2019 @ Siti B
        df_stream = df[
            (df['sits_id'].isin(prequential_ids))
        ].sort_values("timestamp").reset_index(drop=True)
    elif scenario_name == "temporal_spatial":
        # Prequential: 2019 @ Siti B
        df_stream = df[
            (df['timestamp'] >= 365) & 
            (df['sits_id'].isin(prequential_ids))
        ].sort_values("timestamp").reset_index(drop=True)
    else:
        print(f"Scenario '{scenario_name}' non riconosciuto. Salto.")
        return []

    if df_train.empty:
        print(f"Attenzione: Dati di training vuoti per {scenario_name}. Salto.")
        return []
    if df_stream.empty:
        print(f"Attenzione: Dati di streaming vuoti per {scenario_name}. Salto.")
        return []
        
    print(f"Split: {len(df_train)} train samples, {len(df_stream)} prequential samples")
    stream_ts = sorted(df_stream["timestamp"].unique()) # Timestamp per lo stream

    # 2️⃣ Schema
    feature_cols = [c for c in df.columns if c.startswith('emb_')]
    if not feature_cols:
        print("Errore: Nessuna colonna 'emb_' trovata. Salto.")
        return []
        
    schema = Schema.from_custom(
        feature_names=feature_cols,
        target_attribute_name=LABEL_NAME,
        values_for_class_label=list(range(len(CLASS_NAMES)))
    )

    run_name_base = os.path.basename(csv_path).replace(".csv", "")
    results = []

    # 3️⃣ Loop
    for model_name, model_class in tqdm(MODELS.items(), desc=f"Models ({scenario_name})", leave=False):
        
        run_suffix = "adapt" if ADAPT_ON_STREAM else "test_only"
        run_name = f"{run_name_base}_{scenario_name}_{model_name}_{run_suffix}"
        run = wandb.init(
            project=PROJECT_NAME,
            name=run_name, 
            config={
                "embedding_file": csv_path, 
                "model": model_name,
                "scenario": scenario_name,
                "train_sites": train_ids,
                "prequential_sites": prequential_ids,
                "train_samples": len(df_train),
                "stream_samples": len(df_stream),
                "adaptation": ADAPT_ON_STREAM,
            },
            reinit=True
        )

        try:
            model = model_class(schema)
            std_eval = ClassificationEvaluator(schema=schema, window_size=1000)
            change_eval = StreamingChangeEvaluator(num_classes=NUM_CLASSES)

            # 4️⃣ Initial training (2018 + 2019 @ Siti A)
            for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc=f"Initial Train ({model_name})", leave=False):
                y_true = int(row[LABEL_NAME])
                X = np.array([row[c] for c in feature_cols], dtype=float)
                instance = LabeledInstance.from_array(schema, x=X, y_index=y_true)
                model.train(instance)

            # 5️⃣ Prequential test (Stream @ Siti B)
            pbar = tqdm(stream_ts, desc=f"Prequential ({model_name})", leave=False)
            for ts in pbar:
                df_month = df_stream[df_stream["timestamp"] == ts]
                if df_month.empty:
                    continue
                
                print(f"Testing timestamp: {ts} with {len(df_month)} instances")
                
                # Lists for per-timestamp confusion matrices
                y_true_list = []
                y_pred_list = []
                change_true_list = []
                change_pred_list = []
                sc_true_list = []
                sc_pred_list = []

                for _, row in df_month.iterrows():
                    y_true = int(row[LABEL_NAME])
                    X = np.array([row[c] for c in feature_cols], dtype=float)
                    instance = LabeledInstance.from_array(schema, x=X, y_index=y_true)

                    y_pred = int(model.predict(instance))
                    std_eval.update(y_true, y_pred)
                    
                    y_true_list.append(y_true)
                    y_pred_list.append(y_pred)
                    last_state = getattr(change_eval, "_last_state", {})
                    if row[PATCH_ID_COLUMN_NAME] in last_state:
                        y_prev, y_pred_prev = last_state[row[PATCH_ID_COLUMN_NAME]]
                        
                        # 2. Compare current (t) vs previous (t-1)
                        gt_change = 1 if y_true != y_prev else 0
                        pred_change = 1 if y_pred != y_pred_prev else 0
                        
                        # 3. Append to lists for plotting
                        change_true_list.append(gt_change)
                        change_pred_list.append(pred_change)

                        if gt_change == 1:
                            sc_true_list.append(y_true)
                            sc_pred_list.append(y_pred)
                    
                    change_eval.update(row[PATCH_ID_COLUMN_NAME], y_true, y_pred)
                
                if ADAPT_ON_STREAM:
                    print(f"Training on month {ts}...")
                    for _, row in df_month.iterrows():
                            y_true = int(row[LABEL_NAME])
                            X = np.array([row[c] for c in feature_cols], dtype=float)
                            instance = LabeledInstance.from_array(schema, x=X, y_index=y_true)
                            model.train(instance)

                # Log metrics
                metrics = change_eval.compute()
                log_data = {**metrics,
                            "std_accuracy": std_eval.accuracy(),
                            "std_precision": std_eval.precision(),
                            "std_recall": std_eval.recall(),
                            "std_f1": std_eval.f1_score(),
                            "std_kappa": std_eval.kappa(),
                            "std_kappa_m": std_eval.kappa_m(),
                            "std_kappa_t": std_eval.kappa_t(),
                           }
                
                 # 1. Classification CM (per-timestamp)
                if y_true_list:
                    fig_cm = plot_confusion_matrix_image(y_true_list, y_pred_list, CLASS_NAMES, title=f"Confusion Matrix (ts={ts})")
                    log_data["Classification CM (per-timestamp)"] = wandb.Image(fig_cm, caption=f"Classification CM ts={ts}")
                    plt.close(fig_cm)

                # 2. Binary Change CM (per-timestamp)
                if change_true_list:
                    fig_change = plot_confusion_matrix_image(change_true_list, change_pred_list, ["no_change", "change"], title=f"Change Confusion Matrix (ts={ts})")
                    log_data["Change CM (per-timestamp)"] = wandb.Image(fig_change, caption=f"Change Confusion Matrix ts={ts}")
                    plt.close(fig_change)

                # 3. Semantic Change CM (per-timestamp)
                if sc_true_list:
                    fig_sc = plot_confusion_matrix_image(sc_true_list, sc_pred_list, CLASS_NAMES, title=f"Semantic Change CM (ts={ts})")
                    log_data["Semantic Change CM (per-timestamp)"] = wandb.Image(fig_sc, caption=f"Semantic Change CM ts={ts}")
                    plt.close(fig_sc)
                
                wandb.log(log_data, step=ts) 

                pbar.set_postfix({
                    "acc": f"{std_eval.accuracy():.3f}",
                    "scs": f"{metrics.get('scs', 0.0):.3f}",
                    "miou": f"{metrics.get('miou', 0.0):.3f}"
                })

            # 6️⃣ Final metrics
            final_metrics = {
                "scenario": scenario_name,
                "embedding": run_name_base,
                "model": model_name,
                "accuracy": std_eval.accuracy(),
                "precision": std_eval.precision(),
                "recall": std_eval.recall(),
                "f1": std_eval.f1_score(),
                "kappa": std_eval.kappa(),
                "kappa_m": std_eval.kappa_m(),
                "kappa_t": std_eval.kappa_t(),
                **change_eval.compute(),
            }
            results.append(final_metrics)

        except Exception as e:
            print(f"🚨 ERROR running {run_name}: {e}")
        
        finally:
            run.finish()

    return results

# ---------------- Master loop over all embeddings & scenarios ----------------
all_results_in_memory = []
all_files = [file for file in sorted(os.listdir(PROCESSED_DIR)) if file.endswith(".csv")]

OUTPUT_CSV_FILE = "domain_shift_results.csv" # Nome file di output
print(f"Saving incremental results to {OUTPUT_CSV_FILE}")

# Load the JSON file with splits
try:
    with open(SPLITS_JSON_PATH) as f:
        sits_splits = json.load(f)
    print(f"Caricato {SPLITS_JSON_PATH}")
except FileNotFoundError:
    print(f"ERROR: Split file not found: {SPLITS_JSON_PATH}")
    exit()

# Loop sui file CSV
for file in tqdm(all_files, desc="Processing Embedding Files"):
    file_path = os.path.join(PROCESSED_DIR, file)
#file_path ="/Users/chiaranguyen/Desktop/SitsSCD/stream/embeddings/embeddings_dino_sat493m.csv"
    print(f"\n=== Loading {file_path} ===")
    df_full = pd.read_csv(file_path)

    # Loop sugli Scenari
    for scenario in tqdm(SCENARIOS_TO_RUN, desc=f"Scenarios ({os.path.basename(file_path)})", leave=False):
        
        # Esegui l'esperimento per questo file E questo scenario
        res = run_scenario_experiment(
            df_full, 
            sits_splits, 
            scenario, 
            file_path
        )
        
        if res:
            df_batch = pd.DataFrame(res)
            write_header = not os.path.exists(OUTPUT_CSV_FILE)
            df_batch.to_csv(
                OUTPUT_CSV_FILE, 
                mode='a',
                header=write_header, 
                index=False
            )
            all_results_in_memory.extend(res)

# ---------------- Save combined results ----------------
print(f"\nAll results saved incrementally to {OUTPUT_CSV_FILE}")
print("Logging summary table to WandB...")

if all_results_in_memory:
    df_all = pd.DataFrame(all_results_in_memory)
    
    df_all = df_all.sort_values(by=["scenario", "accuracy"], ascending=[True, False])
    
    wandb.init(project=PROJECT_NAME, name="all_domain_shift_summary", reinit=True)
    wandb.log({"all_domain_shift_results": wandb.Table(dataframe=df_all)})
    wandb.finish()
else:
    print("No results were generated to log to WandB.")

print("\n✅ All domain shift evaluations completed.")