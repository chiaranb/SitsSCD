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
from collections import defaultdict
import csv
import os
import json 
from sklearn import linear_model, multiclass

from metrics import StreamingChangeEvaluator, NUM_CLASSES, CLASS_NAMES
from utils import plot_confusion_matrix_image, init_csv_files, plot_patch_timeline

# ---------------- Configuration ----------------
wandb.login()
PROJECT_NAME = "capymoa-streaming-spatial-final" 

ADAPT_ON_STREAM = True
LOG_CONFUSION_MATRICES = True

PROCESSED_DIR = "/Users/chiaranguyen/Desktop/SitsSCD/stream/embeddings"
SPLITS_JSON_PATH = "/Users/chiaranguyen/Desktop/SitsSCD/stream/split.json"
SCENARIOS_TO_RUN = ["temporal_spatial"]

PATCH_ID_COLUMN_NAME = "patch_id"
LABEL_NAME = "label"
OTHER_FEATURES = ["sits_id", "timestamp"]
RANDOM_SEED = 42

# ---------------- Define models ----------------
MODELS = {
    #"SAMkNN": {
    #    "class": SAMkNN,
    #    "params": {
    #        "random_seed": RANDOM_SEED,       
    #        "k": 2, 
    #        "min_stm_size": 90,
    #        "relative_ltm_size": 0.7,
    #        "limit": 20000,
    #    }
    #},
        "HoeffdingTree": {
        "class": HoeffdingTree,
        "params": {
            "random_seed": RANDOM_SEED,
        }
    },
    "SGDClassifier": {
        "class": SGDClassifier,
        "params": {
            "random_seed": RANDOM_SEED,
            "loss": "log_loss",
        }
    },
    "SGDClassifier_SVM": {
        "class": SGDClassifier,
        "params": {
            "random_seed": RANDOM_SEED,
            "loss": "hinge",
        }
    }
}

# ---------------- CSV LOG FILES ----------------
MISCLASS_FILE = "misclassified_log_test.csv"
WRONG_BC_FILE = "wrong_bc_log_test.csv"
WRONG_SC_FILE = "wrong_sc_log_test.csv"

# Initialize CSV log files
init_csv_files(MISCLASS_FILE, WRONG_BC_FILE, WRONG_SC_FILE)

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
    
    # Count class distribution in training set
    train_class_counts = (
        df_train[LABEL_NAME]
        .value_counts()
        .reindex(range(len(CLASS_NAMES)), fill_value=0)
        .to_dict()
    )
    print("Training class distribution:", train_class_counts)
    
    # Count class distribution in streaming set
    stream_class_counts = (
        df_stream[LABEL_NAME]
        .value_counts()
        .reindex(range(len(CLASS_NAMES)), fill_value=0)
        .to_dict()
    )
    print("Streaming class distribution:", stream_class_counts)
    
    # 2️⃣ Schema
    feature_cols = [c for c in df.columns if c not in [PATCH_ID_COLUMN_NAME, LABEL_NAME, *OTHER_FEATURES]]
    schema = Schema.from_custom(
        feature_names=feature_cols,
        target_attribute_name=LABEL_NAME,
        values_for_class_label=list(range(len(CLASS_NAMES)))
    )

    run_name_base = os.path.basename(csv_path).replace(".csv", "")
    results = []

    # 3️⃣ Loop
    for model_name, model_cfg in tqdm(MODELS.items(), desc=f"Models ({scenario_name})", leave=False):
        run_suffix = "adapt" if ADAPT_ON_STREAM else "test_only"
        run_name = f"{run_name_base}_{scenario_name}_{model_name}_{run_suffix}"
        run = wandb.init(
            project=PROJECT_NAME,
            name=run_name, 
            config={
                "embedding_file": csv_path, 
                "model": model_name,
                "adaptation": ADAPT_ON_STREAM,
                "scenario": scenario_name,
                "train_sites": train_ids,
                "prequential_sites": prequential_ids,
                "train_samples": len(df_train),
                "stream_samples": len(df_stream),
                "train_class_counts": train_class_counts,
                "stream_class_counts": stream_class_counts,
                **model_cfg["params"]
            },
            reinit=True
        )
        run_id = run.id

        try:
            model = model_cfg["class"](schema=schema, **model_cfg["params"])
            #model = model_cfg(schema)    
            std_eval_cum = ClassificationEvaluator(schema=schema)
            change_eval_cum = StreamingChangeEvaluator(num_classes=NUM_CLASSES)

            # Cumulative lists for confusion matrices
            y_true_cum_list, y_pred_cum_list = [], []
            change_true_cum_list, change_pred_cum_list = [], [] 
            sc_true_cum_list, sc_pred_cum_list = [], []
            
            # ---- timeline storage ----
            patch_timelines = defaultdict(lambda: {
                "timestamps": [], "y_true": [], "y_pred": []
            })
            
            # ---- BC / SC patch sets ----
            bc_patches = set()
            sc_patches = set()
            
            print(f"\nInitial training for {model_name}...")
            # 4️⃣ Initial training (2018 + 2019 @ Siti A)
            for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc=f"Initial Train ({model_name})", leave=False):
                y_true = int(row[LABEL_NAME])
                X = np.array([row[c] for c in feature_cols], dtype=float)
                instance = LabeledInstance.from_array(schema, x=X, y_index=y_true)
                model.train(instance)
            print("Initial training completed.")

            # 5️⃣ Prequential test (Stream @ Siti B)
            pbar = tqdm(stream_ts, desc=f"Prequential ({model_name})", leave=False)
            for ts in pbar:
                df_month = df_stream[df_stream["timestamp"] == ts]
                if df_month.empty:
                    continue
                
                # Monthly evaluators
                std_eval_month = ClassificationEvaluator(schema=schema)
                change_eval_month = StreamingChangeEvaluator(num_classes=NUM_CLASSES)
                
                print(f"Testing timestamp: {ts} with {len(df_month)} instances")
                
                # Monthly lists for confusion matrices
                y_true_month_list, y_pred_month_list = [], []
                change_true_month_list, change_pred_month_list = [], []
                sc_true_month_list, sc_pred_month_list = [], []
                misclassified, wrong_bc, wrong_sc = [], [], []

                for _, row in df_month.iterrows():
                    patch_id = row[PATCH_ID_COLUMN_NAME]
                    sits_id = row["sits_id"]
                    timestamp = row["timestamp"]
                    y_true = int(row[LABEL_NAME])
                    X = np.array([row[c] for c in feature_cols], dtype=float)
                    instance = LabeledInstance.from_array(schema, x=X, y_index=y_true)

                    y_pred = int(model.predict(instance))
                    
                    # Update standard evaluator
                    std_eval_cum.update(y_true, y_pred)
                    std_eval_month.update(y_true, y_pred)
                    
                    # timeline
                    patch_timelines[patch_id]["timestamps"].append(ts)
                    patch_timelines[patch_id]["y_true"].append(y_true)
                    patch_timelines[patch_id]["y_pred"].append(y_pred)
                    
                    # Append values for the standard classification CM
                    y_true_month_list.append(y_true)
                    y_pred_month_list.append(y_pred)
                    
                    y_true_cum_list.append(y_true)
                    y_pred_cum_list.append(y_pred)
                    
                    if y_pred != y_true:
                        misclassified.append({
                            "sits_id": sits_id,
                            "patch_id": patch_id,
                            "timestamp": timestamp,
                            "y_true": y_true,
                            "y_pred": y_pred
                        })
                        
                    last_state_cum = getattr(change_eval_cum, "_last_state", {})
                    if row[PATCH_ID_COLUMN_NAME] in last_state_cum:
                        y_prev, y_pred_prev = last_state_cum[row[PATCH_ID_COLUMN_NAME]]
                        
                        # 2. Compare current (t) vs previous (t-1)
                        gt_change = 1 if y_true != y_prev else 0
                        pred_change = 1 if y_pred != y_pred_prev else 0
                        
                        # 3. Append to lists for plotting
                        change_true_cum_list.append(gt_change)
                        change_pred_cum_list.append(pred_change)
                        
                        change_true_month_list.append(gt_change)
                        change_pred_month_list.append(pred_change)

                        if gt_change == 1:
                            sc_true_cum_list.append(y_true)
                            sc_pred_cum_list.append(y_pred)
                            
                            sc_true_month_list.append(y_true)
                            sc_pred_month_list.append(y_pred)
                        
                        if gt_change != pred_change:
                            bc_patches.add(patch_id)
                            wrong_bc.append({
                                "sits_id": sits_id,
                                "patch_id": patch_id,
                                "timestamp": timestamp,
                                "y_true_t-1": y_prev,
                                "y_true_t": y_true,
                                "y_pred_t-1": y_pred_prev,
                                "y_pred_t": y_pred
                            })
                            
                        if gt_change == 1 and y_pred != y_true:
                            sc_patches.add(patch_id)
                            wrong_sc.append({
                                "sits_id": sits_id,
                                "patch_id": patch_id,
                                "timestamp": timestamp,
                                "y_true_t-1": y_prev,
                                "y_true_t": y_true,
                                "y_pred_t-1": y_pred_prev,
                                "y_pred_t": y_pred
                            })
                    
                    change_eval_cum.update(row[PATCH_ID_COLUMN_NAME], y_true, y_pred)
                    change_eval_month.update(row[PATCH_ID_COLUMN_NAME], y_true, y_pred)
                    
                if ADAPT_ON_STREAM:
                    print(f"Training on month {ts}...")
                    for _, row in df_month.iterrows():
                            y_true = int(row[LABEL_NAME])
                            X = np.array([row[c] for c in feature_cols], dtype=float)
                            instance = LabeledInstance.from_array(schema, x=X, y_index=y_true)
                            model.train(instance)

                with open(MISCLASS_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    for d in misclassified:
                        writer.writerow([
                            run_id, ts,
                            d["sits_id"], d["patch_id"],
                            d["y_true"], d["y_pred"]
                        ])

                with open(WRONG_BC_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    for d in wrong_bc:
                        writer.writerow([
                            run_id, ts,
                            d["sits_id"], d["patch_id"],
                            d["y_true_t-1"], d["y_true_t"],
                            d["y_pred_t-1"], d["y_pred_t"]
                        ])

                with open(WRONG_SC_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    for d in wrong_sc:
                        writer.writerow([
                            run_id, ts,
                            d["sits_id"], d["patch_id"],
                            d["y_true_t-1"], d["y_true_t"], 
                            d["y_pred_t-1"], d["y_pred_t"]
                        ])
                        
                # Get cumulative metrics
                metrics_cum = change_eval_cum.compute(prefix="")
                log_cum_data = {**metrics_cum,
                            "accuracy": std_eval_cum.accuracy(),
                            "precision": std_eval_cum.precision(),
                            "recall": std_eval_cum.recall(),
                            "f1": std_eval_cum.f1_score(),
                            "kappa": std_eval_cum.kappa(),
                            "kappa_m": std_eval_cum.kappa_m(),
                            "kappa_t": std_eval_cum.kappa_t(),
                           }
                
                # Get monthly metrics
                metrics_month = change_eval_month.compute(prefix="month/")
                log_month_data = {**metrics_month,
                            "month/accuracy": std_eval_month.accuracy(),
                            "month/precision": std_eval_month.precision(),
                            "month/recall": std_eval_month.recall(),
                            "month/f1": std_eval_month.f1_score(),
                            "month/kappa": std_eval_month.kappa(),
                            "month/kappa_m": std_eval_month.kappa_m(),
                            "month/kappa_t": std_eval_month.kappa_t(),
                            }
                
                if LOG_CONFUSION_MATRICES:
                    # 1. Classification CM (per-timestamp)
                    if y_true_month_list:
                        fig_cm = plot_confusion_matrix_image(y_true_month_list, y_pred_month_list, CLASS_NAMES, title=f"Confusion Matrix (ts={ts})")
                        log_month_data["Classification CM (per-timestamp)"] = wandb.Image(fig_cm, caption=f"Classification CM ts={ts}")
                        plt.close(fig_cm)
                        
                    if y_true_cum_list:
                        fig_cm_cum = plot_confusion_matrix_image(y_true_cum_list, y_pred_cum_list, CLASS_NAMES, title=f"Cumulative Confusion Matrix (up to ts={ts})")
                        log_cum_data["Cumulative Classification CM"] = wandb.Image(fig_cm_cum, caption=f"Cumulative Classification CM up to ts={ts}")
                        plt.close(fig_cm_cum)

                    # 2. Binary Change CM (per-timestamp)
                    if change_true_month_list:
                        fig_change = plot_confusion_matrix_image(change_true_month_list, change_pred_month_list, ["no_change", "change"], title=f"Change Confusion Matrix (ts={ts})")
                        log_month_data["Change CM (per-timestamp)"] = wandb.Image(fig_change, caption=f"Change Confusion Matrix ts={ts}")
                        plt.close(fig_change)
                    
                    if change_true_cum_list:
                        fig_change_cum = plot_confusion_matrix_image(change_true_cum_list, change_pred_cum_list, ["no_change", "change"], title=f"Cumulative Change Confusion Matrix (up to ts={ts})")
                        log_cum_data["Cumulative Change CM"] = wandb.Image(fig_change_cum, caption=f"Cumulative Change CM up to ts={ts}")
                        plt.close(fig_change_cum)

                    # 3. Semantic Change CM (per-timestamp)
                    if sc_true_month_list:
                        fig_sc = plot_confusion_matrix_image(sc_true_month_list, sc_pred_month_list, CLASS_NAMES, title=f"Semantic Change CM (ts={ts})")
                        log_month_data["Semantic Change CM (per-timestamp)"] = wandb.Image(fig_sc, caption=f"Semantic Change CM ts={ts}")
                        plt.close(fig_sc)
                        
                    if sc_true_cum_list:
                        fig_sc_cum = plot_confusion_matrix_image(sc_true_cum_list, sc_pred_cum_list, CLASS_NAMES, title=f"Cumulative Semantic Change CM (up to ts={ts})")
                        log_cum_data["Cumulative Semantic Change CM"] = wandb.Image(fig_sc_cum, caption=f"Cumulative Semantic Change CM up to ts={ts}")
                        plt.close(fig_sc_cum)
                    
                wandb.log(log_cum_data, step=ts) 
                wandb.log(log_month_data, step=ts)

                pbar.set_postfix({
                    "acc": f"{std_eval_cum.accuracy():.3f}",
                    "scs": f"{metrics_cum.get('scs', 0.0):.3f}",
                    "miou": f"{metrics_cum.get('miou', 0.0):.3f}"
                })
            
            # Timeline plots
            patches_to_log = bc_patches.union(sc_patches)
            print(f"Logging {len(patches_to_log)} BC/SC patch timelines")

            for patch in patches_to_log:
                # Convert patch to int if possible
                patch = int(patch)
                seq = patch_timelines.get(patch)
                if seq is None or len(seq["timestamps"]) < 2:
                    continue

                fig = plot_patch_timeline(
                    seq["timestamps"], seq["y_true"], seq["y_pred"],
                    CLASS_NAMES, f"Patch {patch}"
                )
                wandb.log({f"timeline/patch_{patch}": wandb.Image(fig)})
                plt.close(fig)

            # 6️⃣ Final metrics
            final_metrics = {
                "embedding": run_name_base,
                "model": model_name,
                "accuracy": std_eval_cum.accuracy(),
                "precision": std_eval_cum.precision(),
                "recall": std_eval_cum.recall(),
                "f1": std_eval_cum.f1_score(),
                "kappa": std_eval_cum.kappa(),
                "kappa_m": std_eval_cum.kappa_m(),
                "kappa_t": std_eval_cum.kappa_t(),
                **change_eval_cum.compute(),
            }
            results.append(final_metrics)

        except Exception as e:
            print(f"🚨 ERROR running {run_name}: {e}")
        
        finally:
            artifact = wandb.Artifact(
                name="error_logs",
                type="dataset",
                description="Misclassified, wrong BC and wrong SC logs"
            )

            artifact.add_file(MISCLASS_FILE)
            artifact.add_file(WRONG_BC_FILE)
            artifact.add_file(WRONG_SC_FILE)

            wandb.log_artifact(artifact)
            
            run.finish()

    return results

# ---------------- Master loop over all embeddings & scenarios ----------------
all_results_in_memory = []
all_files = [file for file in sorted(os.listdir(PROCESSED_DIR)) if file.endswith(".csv") and not file.startswith("._")]

OUTPUT_CSV_FILE = "final_spatial_results.csv" 
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
#for file in tqdm(all_files, desc="Processing Embedding Files"):
#    file_path = os.path.join(PROCESSED_DIR, file)
file_path = "/Volumes/PSSD T7/SitsSCD/processed_embeddings/DINO/Proj_Scale/PCA/emb_dino_sat493m_pca256.csv"
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
print("\n✅ All domain shift evaluations completed.")