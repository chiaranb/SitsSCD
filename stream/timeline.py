import wandb
from capymoa.classifier import SAMkNN
from capymoa.evaluation import ClassificationEvaluator
from tqdm import tqdm
import pandas as pd
import numpy as np
from capymoa.instance import LabeledInstance
from capymoa.stream import Schema
import os
import csv
import matplotlib.pyplot as plt
from collections import defaultdict

from metrics import StreamingChangeEvaluator, NUM_CLASSES, CLASS_NAMES
from utils import plot_confusion_matrix_image

# ---------------- Configuration ----------------
wandb.login()
PROJECT_NAME = "capymoa-streaming"

ADAPT_ON_STREAM = False
LOG_CONFUSION_MATRICES = True

PROCESSED_DIR = "/Volumes/PSSD T7/SitsSCD/processed_embeddings/DINO/Proj_Scale"
PATCH_ID_COLUMN_NAME = "patch_id"
LABEL_NAME = "label"
OTHER_FEATURES = ["sits_id", "timestamp"]

DIFF_MONTHS = False
MONTHS_PER_YEAR = 12
MONTHS_TRAIN = 6 if DIFF_MONTHS else None
MONTHS_TEST = 18 if DIFF_MONTHS else None
RANDOM_SEED = 42

# ---------------- Define models ----------------
MODELS = {
    "SAMKNN": lambda schema: SAMkNN(
        schema=schema,
        random_seed=RANDOM_SEED,
        min_stm_size=20,
        relative_ltm_size=0.3,
        k=4
    )
}

# ---------------- CSV LOG FILES ----------------
MISCLASS_FILE = "misclassified_log_test.csv"
WRONG_BC_FILE = "wrong_bc_log_test.csv"
WRONG_SC_FILE = "wrong_sc_log_test.csv"

def init_csv_files():
    if not os.path.exists(MISCLASS_FILE):
        with open(MISCLASS_FILE, "w", newline="") as f:
            csv.writer(f).writerow(
                ["run_id","timestamp","sits_id","patch_id","y_true","y_pred"]
            )

    if not os.path.exists(WRONG_BC_FILE):
        with open(WRONG_BC_FILE, "w", newline="") as f:
            csv.writer(f).writerow(
                ["run_id","timestamp","sits_id","patch_id",
                 "y_true_t-1","y_true_t","y_pred_t-1","y_pred_t"]
            )

    if not os.path.exists(WRONG_SC_FILE):
        with open(WRONG_SC_FILE, "w", newline="") as f:
            csv.writer(f).writerow(
                ["run_id","timestamp","sits_id","patch_id",
                 "y_true_t-1","y_true_t","y_pred_t-1","y_pred_t"]
            )

# ---------------- Timeline plot ----------------
def plot_patch_timeline(timestamps, y_true_seq, y_pred_seq, class_names, title):
    x = list(range(len(timestamps)))

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(9, 3))

    ax.plot(x, y_true_seq, "s-", label="GT")
    ax.plot(x, y_pred_seq, "o--", label="Pred")

    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xticks(x)
    ax.set_xticklabels(
        month_labels[:len(x)],
        fontsize=7
    )
    ax.grid(True, axis="x", which="major", linestyle="--", alpha=0.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    ax.set_xlabel("Month")
    ax.set_ylabel("Class")
    ax.set_title(title)
    ax.legend(frameon=False)

    plt.tight_layout()
    return fig

# ---------------- Helper: prequential loop ----------------
def run_prequential_experiment(csv_path: str):
    print(f"\n=== Loading {csv_path} ===")
    df = pd.read_csv(csv_path).sort_values("timestamp").reset_index(drop=True)
    unique_ts = sorted(df["timestamp"].unique())

    train_ts = unique_ts[:MONTHS_TRAIN] if MONTHS_TRAIN else unique_ts[:MONTHS_PER_YEAR]
    stream_ts = (
        unique_ts[MONTHS_TRAIN:MONTHS_TRAIN+MONTHS_TEST]
        if MONTHS_TEST else unique_ts[MONTHS_PER_YEAR:]
    )

    df_train = df[df["timestamp"].isin(train_ts)]
    df_stream = df[df["timestamp"].isin(stream_ts)]

    feature_cols = [
        c for c in df.columns
        if c not in [PATCH_ID_COLUMN_NAME, LABEL_NAME, *OTHER_FEATURES]
    ]

    schema = Schema.from_custom(
        feature_names=feature_cols,
        target_attribute_name=LABEL_NAME,
        values_for_class_label=list(range(len(CLASS_NAMES)))
    )

    run_name_base = os.path.basename(csv_path).replace(".csv", "")
    init_csv_files()
    results = []

    for model_name, model_class in tqdm(MODELS.items(), desc=f"Models ({run_name_base})", leave=False):
        run_name = f"{run_name_base}_{model_name}_{'adapt' if ADAPT_ON_STREAM else 'test_only'}"
        run = wandb.init(
            project=PROJECT_NAME,
            name=run_name,
            config={
                "embedding_file": csv_path,
                "model": model_name,
                "adaptation": ADAPT_ON_STREAM
            },
            reinit=True
        )
        run_id = run.id

        try:
            model = model_class(schema)
            std_eval = ClassificationEvaluator(schema=schema, window_size=1000)
            change_eval = StreamingChangeEvaluator(num_classes=NUM_CLASSES)

            # ---- timeline storage ----
            patch_timelines = defaultdict(lambda: {
                "timestamps": [], "y_true": [], "y_pred": []
            })

            # ---- BC / SC patch sets ----
            bc_patches = set()
            sc_patches = set()

            # ---- Initial training ----
            print(f"\nInitial training for {model_name}...")
            for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Initial Train"):
                X = np.array([row[c] for c in feature_cols], float)
                y = int(row[LABEL_NAME])
                model.train(LabeledInstance.from_array(schema, x=X, y_index=y))
            print("Initial training completed.")

            # ---- Streaming ----
            progress = tqdm(stream_ts, desc=f"Prequential ({model_name})", leave=False)
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
                    patch = row[PATCH_ID_COLUMN_NAME]
                    sits_id = row["sits_id"]
                    y_true = int(row[LABEL_NAME])
                    X = np.array([row[c] for c in feature_cols], float)
                    inst = LabeledInstance.from_array(schema, x=X, y_index=y_true)

                    y_pred = int(model.predict(inst))
                    std_eval.update(y_true, y_pred)

                    # timeline
                    patch_timelines[patch]["timestamps"].append(ts)
                    patch_timelines[patch]["y_true"].append(y_true)
                    patch_timelines[patch]["y_pred"].append(y_pred)

                    y_true_list.append(y_true)
                    y_pred_list.append(y_pred)

                    if y_pred != y_true:
                        misclassified.append([run_id, ts, sits_id, patch, y_true, y_pred])

                    last_state = getattr(change_eval, "_last_state", {})
                    if patch in last_state:
                        y_prev, y_pred_prev = last_state[patch]
                        gt_change = y_true != y_prev
                        pred_change = y_pred != y_pred_prev

                        change_true_list.append(int(gt_change))
                        change_pred_list.append(int(pred_change))

                        if gt_change:
                            sc_true_list.append(y_true)
                            sc_pred_list.append(y_pred)

                        if gt_change != pred_change:
                            bc_patches.add(patch)
                            wrong_bc.append(
                                [run_id, ts, sits_id, patch, y_prev, y_true, y_pred_prev, y_pred]
                            )

                        if gt_change and y_pred != y_true:
                            sc_patches.add(patch)
                            wrong_sc.append(
                                [run_id, ts, sits_id, patch, y_prev, y_true, y_pred_prev, y_pred]
                            )

                    change_eval.update(patch, y_true, y_pred)

                if ADAPT_ON_STREAM:
                    print(f"Training on month {ts}...")
                    for _, row in df_month.iterrows():
                        X = np.array([row[c] for c in feature_cols], float)
                        y = int(row[LABEL_NAME])
                        model.train(LabeledInstance.from_array(schema, x=X, y_index=y))

                # ---- write CSV logs ----
                with open(MISCLASS_FILE, "a", newline="") as f:
                    csv.writer(f).writerows(misclassified)
                with open(WRONG_BC_FILE, "a", newline="") as f:
                    csv.writer(f).writerows(wrong_bc)
                with open(WRONG_SC_FILE, "a", newline="") as f:
                    csv.writer(f).writerows(wrong_sc)

                # ---- metrics + CMs ----
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

                if LOG_CONFUSION_MATRICES:
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
                
                progress.set_postfix({
                    "acc": f"{std_eval.accuracy():.3f}",
                    "scs": f"{metrics.get('scs', 0.0):.3f}",
                    "miou": f"{metrics.get('miou', 0.0):.3f}"
                })

            # ---- LOG TIMELINES ----
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
                wandb.log({f"timelines_bc_sc/patch_{patch}": wandb.Image(fig)})
                plt.close(fig)

            final_metrics = {
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
            print(f"🚨 ERROR running model {model_name} on {run_name_base}: {e}")
            print("Skipping to next model...")

        finally:
            artifact = wandb.Artifact("error_logs", type="dataset")
            artifact.add_file(MISCLASS_FILE)
            artifact.add_file(WRONG_BC_FILE)
            artifact.add_file(WRONG_SC_FILE)
            wandb.log_artifact(artifact)
            run.finish()

    return results

# ---------------- Master loop over all embeddings ----------------
all_results_in_memory = []
all_files = [
    file for file in sorted(os.listdir(PROCESSED_DIR))
    if file.endswith(".csv") and not file.startswith("._")
]

OUTPUT_CSV_FILE = "search_results_all_embeddings_preprocessing.csv"
print(f"Saving incremental results to {OUTPUT_CSV_FILE}")

# ---------------- Run ----------------
#for file in tqdm(all_files, desc="Processing Embedding Files"):
    #file_path = os.path.join(PROCESSED_DIR, file)
file_path = "/Volumes/PSSD T7/SitsSCD/processed_embeddings/DINO/Proj_Scale/PCA/emb_dino_sat493m_pca256_randomized.csv"
res = run_prequential_experiment(file_path)
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
    wandb.init(project=PROJECT_NAME, name="all_embedding_summary", reinit=True)
    wandb.log({"all_embedding_results": wandb.Table(dataframe=df_all)})
    wandb.finish()
else:
    print("No results were generated to log to WandB.")

print("\nCompleted.")