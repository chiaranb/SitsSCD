import wandb
from capymoa.classifier import SAMkNN
from capymoa.evaluation import ClassificationEvaluator
from tqdm import tqdm
import pandas as pd
import numpy as np
from capymoa.instance import LabeledInstance
from capymoa.stream import Schema
from capymoa.base import SKClassifier
from sklearn import linear_model

import os
import matplotlib.pyplot as plt
from metrics import StreamingChangeEvaluator, NUM_CLASSES, CLASS_NAMES
from utils import plot_confusion_matrix_image
import csv

# ---------------- Configuration ----------------
wandb.login()
PROJECT_NAME = "capymoa-streaming-ensemble"

ADAPT_ON_STREAM = True
PROCESSED_DIR = "/Volumes/PSSD T7/SitsSCD/processed_embeddings/DINO/Proj_Scale"
PATCH_ID_COLUMN_NAME = "patch_id"
LABEL_NAME = "label"
OTHER_FEATURES = ["sits_id", "timestamp"]
DIFF_MONTHS = False
MONTHS_PER_YEAR = 12
MONTHS_TRAIN = 6 if DIFF_MONTHS else None
MONTHS_TEST = 18 if DIFF_MONTHS else None
RANDOM_SEED = 42

# ---------------- Define ensemble components ----------------
ENSEMBLE_MODELS = {
    "samknn_k4": lambda schema: SAMkNN(
        schema=schema,
        random_seed=RANDOM_SEED,
        min_stm_size=20,
        relative_ltm_size=0.3,
        k=4,
    ),
    "samknn_k3": lambda schema: SAMkNN(
        schema=schema,
        random_seed=RANDOM_SEED,
        min_stm_size=20,
        relative_ltm_size=0.3,
        k=3,
    ),
}

ENSEMBLE_NAME = "SAMKNN_k4_SAMKNN_k3_gated_rare"

# ---------------- CSV LOG FILES ----------------
MISCLASS_FILE = "misclassified_log_test.csv"
WRONG_BC_FILE = "wrong_bc_log_test.csv"
WRONG_SC_FILE = "wrong_sc_log_test.csv"

def init_csv_files():
    """Create the CSV files if they do not already exist."""
    if not os.path.exists(MISCLASS_FILE):
        with open(MISCLASS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_id", "timestamp",
                "sits_id", "patch_id",
                "y_true", "y_pred"
            ])

    if not os.path.exists(WRONG_BC_FILE):
        with open(WRONG_BC_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_id", "timestamp",
                "sits_id", "patch_id",
                "y_true_t-1", "y_true_t",
                "y_pred_t-1", "y_pred_t"
            ])

    if not os.path.exists(WRONG_SC_FILE):
        with open(WRONG_SC_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_id", "timestamp",
                "sits_id", "patch_id",
                "y_true_t-1", "y_true_t",
                "y_pred_t-1", "y_pred_t"
            ])

# ---------------- Helper: prequential loop ----------------
def run_prequential_experiment(csv_path: str):
    print(f"\n=== Loading {csv_path} ===")
    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    unique_ts = sorted(df["timestamp"].unique())

    # Temporal split
    train_ts = unique_ts[:MONTHS_TRAIN] if MONTHS_TRAIN is not None else unique_ts[:MONTHS_PER_YEAR]
    stream_ts = (
        unique_ts[MONTHS_TRAIN:MONTHS_TRAIN + MONTHS_TEST]
        if MONTHS_TEST is not None
        else unique_ts[MONTHS_PER_YEAR:]
    )

    df_train = df[df["timestamp"].isin(train_ts)]
    df_stream = df[df["timestamp"].isin(stream_ts)]

    feature_cols = [c for c in df.columns if c not in [PATCH_ID_COLUMN_NAME, LABEL_NAME, *OTHER_FEATURES]]
    schema = Schema.from_custom(
        feature_names=feature_cols,
        target_attribute_name=LABEL_NAME,
        values_for_class_label=list(range(len(CLASS_NAMES))),
    )

    # Identify rare classes from the initial training slice
    class_counts = df_train[LABEL_NAME].value_counts()
    K = 3
    rare_classes = set(class_counts.nsmallest(K).index)
    print(f"Identified rare classes: {rare_classes}")

    run_name_base = os.path.basename(csv_path).replace(".csv", "")
    run_suffix = "adapt" if ADAPT_ON_STREAM else "test_only"
    run_name = f"{run_name_base}_{ENSEMBLE_NAME}_{run_suffix}"

    run = wandb.init(
        project=PROJECT_NAME,
        name=run_name,
        config={
            "embedding_file": csv_path,
            "model": ENSEMBLE_NAME,
            "adaptation": ADAPT_ON_STREAM,
            "rare_classes": list(rare_classes),
            "components": list(ENSEMBLE_MODELS.keys()),
        },
        reinit=True,
    )
    run_id = run.id

    results = []
    try:
        models = {name: builder(schema) for name, builder in ENSEMBLE_MODELS.items()}
        std_eval = ClassificationEvaluator(schema=schema, window_size=1000)
        change_eval = StreamingChangeEvaluator(num_classes=NUM_CLASSES)

        # 1) Initial training
        print(f"\nInitial training for ensemble {ENSEMBLE_NAME}...")
        for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Initial Train (ensemble)", leave=False):
            y_true = int(row[LABEL_NAME])
            X = np.array([row[c] for c in feature_cols], dtype=float)
            instance = LabeledInstance.from_array(schema, x=X, y_index=y_true)
            for m in models.values():
                m.train(instance)
        print("Initial training completed.")

        # 2) Prequential evaluation
        progress = tqdm(stream_ts, desc="Prequential (ensemble)", leave=False)

        for ts in progress:
            df_month = df_stream[df_stream["timestamp"] == ts]
            if df_month.empty:
                continue

            print(f"\nTesting month {ts} ({len(df_month)} instances)")

            y_true_list, y_pred_list = [], []
            change_true_list, change_pred_list = [], []
            sc_true_list, sc_pred_list = [], []
            
            misclassified, wrong_bc, wrong_sc = [], [], []

            for _, row in df_month.iterrows():
                patch_id = row[PATCH_ID_COLUMN_NAME]
                sits_id = row["sits_id"]
                timestamp = row["timestamp"]
                y_true = int(row[LABEL_NAME])

                X = np.array([row[c] for c in feature_cols], dtype=float)
                instance = LabeledInstance.from_array(schema, x=X, y_index=y_true)

                # Predictions from components
                y_pred_sam_k4 = int(models["samknn_k4"].predict(instance))
                y_pred_sam_k3 = int(models["samknn_k3"].predict(instance))

                # Gating: use SAMkNN_k3 only when it predicts a rare class
                y_pred = y_pred_sam_k3 if y_pred_sam_k3 in rare_classes else y_pred_sam_k4

                # Standard metrics
                std_eval.update(y_true, y_pred)
                y_true_list.append(y_true)
                y_pred_list.append(y_pred)
                
                if y_pred != y_true:
                    misclassified.append({
                        "sits_id": sits_id,
                        "patch_id": patch_id,
                        "timestamp": timestamp,
                        "y_true": y_true,
                        "y_pred": y_pred
                    })

                # Change metrics
                last_state = getattr(change_eval, "_last_state", {})
                if patch_id in last_state:
                    y_prev, y_pred_prev = last_state[patch_id]

                    gt_change = 1 if y_true != y_prev else 0
                    pred_change = 1 if y_pred != y_pred_prev else 0

                    change_true_list.append(gt_change)
                    change_pred_list.append(pred_change)

                    if gt_change == 1:
                        sc_true_list.append(y_true)
                        sc_pred_list.append(y_pred)
                    
                    if gt_change != pred_change:
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
                        wrong_sc.append({
                            "sits_id": sits_id,
                            "patch_id": patch_id,
                            "timestamp": timestamp,
                            "y_true_t-1": y_prev,
                            "y_true_t": y_true,
                            "y_pred_t-1": y_pred_prev,
                            "y_pred_t": y_pred
                        })

                change_eval.update(patch_id, y_true, y_pred)

            # Optional adaptation on stream
            if ADAPT_ON_STREAM:
                print(f"Training on month {ts}...")
                for _, row in df_month.iterrows():
                    y_true = int(row[LABEL_NAME])
                    X = np.array([row[c] for c in feature_cols], dtype=float)
                    instance = LabeledInstance.from_array(schema, x=X, y_index=y_true)
                    for m in models.values():
                        m.train(instance)
            
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
                        d["y_true_t-1"], d["y_true_t"], d["y_pred_t-1"], d["y_pred_t"]
                    ])

            metrics = change_eval.compute()
            
            log_data = {
                **metrics,
                "std_accuracy": std_eval.accuracy(),
                "std_precision": std_eval.precision(),
                "std_recall": std_eval.recall(),
                "std_f1": std_eval.f1_score(),
                "std_kappa": std_eval.kappa(),
                "std_kappa_m": std_eval.kappa_m(),
                "std_kappa_t": std_eval.kappa_t(),
            }

            # Confusion matrices (per timestamp)
            if y_true_list:
                fig_cm = plot_confusion_matrix_image(
                    y_true_list, y_pred_list, CLASS_NAMES, title=f"Confusion Matrix (ts={ts})"
                )
                log_data["Classification CM (per-timestamp)"] = wandb.Image(fig_cm, caption=f"Classification CM ts={ts}")
                plt.close(fig_cm)

            if change_true_list:
                fig_change = plot_confusion_matrix_image(
                    change_true_list, change_pred_list, ["no_change", "change"],
                    title=f"Change Confusion Matrix (ts={ts})"
                )
                log_data["Change CM (per-timestamp)"] = wandb.Image(fig_change, caption=f"Change CM ts={ts}")
                plt.close(fig_change)

            if sc_true_list:
                fig_sc = plot_confusion_matrix_image(
                    sc_true_list, sc_pred_list, CLASS_NAMES, title=f"Semantic Change CM (ts={ts})"
                )
                log_data["Semantic Change CM (per-timestamp)"] = wandb.Image(fig_sc, caption=f"Semantic Change CM ts={ts}")
                plt.close(fig_sc)

            wandb.log(log_data, step=int(ts))
            
            progress.set_postfix({
                "acc": f"{std_eval.accuracy():.3f}",
                "scs": f"{metrics.get('scs', 0.0):.3f}",
                "miou": f"{metrics.get('miou', 0.0):.3f}",
            })

        # 3) Final metrics
        final_metrics = {
            "embedding": run_name_base,
            "model": ENSEMBLE_NAME,
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
        print(f"🚨 ERROR running ensemble {ENSEMBLE_NAME} on {run_name_base}: {e}")

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


# ---------------- Master loop ----------------
all_results_in_memory = []
all_files = [
    file for file in sorted(os.listdir(PROCESSED_DIR))
    if file.endswith(".csv") and not file.startswith("._")
]

OUTPUT_CSV_FILE = "search_results_all_embeddings_preprocessing.csv"
print(f"Saving incremental results to {OUTPUT_CSV_FILE}")

file_path = "/Volumes/PSSD T7/SitsSCD/processed_embeddings/DINO/Proj_Scale/PCA/emb_dino_sat493m_pca256_randomized_l2.csv"
res = run_prequential_experiment(file_path)

if res:
    df_batch = pd.DataFrame(res)
    write_header = not os.path.exists(OUTPUT_CSV_FILE)
    df_batch.to_csv(OUTPUT_CSV_FILE, mode="a", header=write_header, index=False)
    all_results_in_memory.extend(res)

print(f"\nAll results saved incrementally to {OUTPUT_CSV_FILE}")
print("Logging summary table to WandB...")

if all_results_in_memory:
    df_all = pd.DataFrame(all_results_in_memory)
    wandb.init(project=PROJECT_NAME, name="all_embedding_summary", reinit=True)
    wandb.log({"all_embedding_results": wandb.Table(dataframe=df_all)})
    wandb.finish()
else:
    print("No results were generated to log to WandB.")

print("\n✅ All embedding evaluations completed.")