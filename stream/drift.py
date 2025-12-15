import wandb
from capymoa.classifier import SAMkNN, SGDClassifier
from capymoa.stream.preprocessing import ClassifierPipeline
from capymoa.drift.detectors import ADWIN
from capymoa.evaluation import ClassificationEvaluator
from capymoa.type_alias import LabelIndex
from tqdm import tqdm
import pandas as pd
import numpy as np
from capymoa.instance import LabeledInstance
from capymoa.stream import Schema
import os
import matplotlib.pyplot as plt
from metrics import StreamingChangeEvaluator, NUM_CLASSES, CLASS_NAMES
from utils import plot_confusion_matrix_image

wandb.login()
PROJECT_NAME = "capymoa-streaming"

ADAPT_ON_STREAM = True
PROCESSED_DIR = "/Volumes/PSSD T7/SitsSCD/processed_embeddings/DINO"
PATCH_ID_COLUMN_NAME = "patch_id"
LABEL_NAME = "label"
OTHER_FEATURES = ["sits_id", "timestamp"]
DIFF_MONTHS = False # True 12 months train and 12 months test
MONTHS_PER_YEAR = 12
MONTHS_TRAIN = 6 if DIFF_MONTHS else None
MONTHS_TEST = 18 if DIFF_MONTHS else None
RANDOM_SEED = 42

LOG_IMAGES = True

MODELS = {
    "SAMKNN": lambda schema: SAMkNN(schema=schema, random_seed=RANDOM_SEED, min_stm_size=20, relative_ltm_size=0.3),
    #"SGD_SVM": lambda schema: SGDClassifier(schema=schema, random_seed=RANDOM_SEED)
}

def label_equals_prediction(instance: LabeledInstance, prediction: LabelIndex) -> LabelIndex:
    label = instance.y_index
    return int(label == prediction)

def run_prequential_experiment(csv_path: str):
    print(f"\n=== Loading {csv_path} ===")
    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    unique_ts = sorted(df["timestamp"].unique())

    train_ts = unique_ts[:MONTHS_TRAIN] if MONTHS_TRAIN is not None else unique_ts[:MONTHS_PER_YEAR]
    stream_ts = unique_ts[MONTHS_TRAIN:MONTHS_TRAIN+MONTHS_TEST] if MONTHS_TEST is not None else unique_ts[MONTHS_PER_YEAR:]

    df_train = df[df["timestamp"].isin(train_ts)]
    df_stream = df[df["timestamp"].isin(stream_ts)]

    feature_cols = [c for c in df.columns if c not in [PATCH_ID_COLUMN_NAME, LABEL_NAME, *OTHER_FEATURES]]
    schema = Schema.from_custom(
        feature_names=feature_cols,
        target_attribute_name=LABEL_NAME,
        values_for_class_label=list(range(len(CLASS_NAMES)))
    )

    run_name_base = os.path.basename(csv_path).replace(".csv", "")
    results = []

    for model_name, model_constructor in tqdm(MODELS.items(), desc=f"Models ({run_name_base})", leave=False):
        run_suffix = "adapt" if ADAPT_ON_STREAM else "test_only"
        run_name = f"{run_name_base}_{model_name}_{run_suffix}"

        run = wandb.init(
            project=PROJECT_NAME,
            name=run_name,
            config={
                "embedding_file": csv_path,
                "model": model_name,
                "adaptation": ADAPT_ON_STREAM,
            },
            reinit=True
        )

        try:
            learner = model_constructor(schema)
            drift_detector = ADWIN()
            
            model = (
                ClassifierPipeline()
                .add_classifier(learner)
                .add_drift_detector(
                    drift_detector, get_drift_detector_input_func=label_equals_prediction
                )
            )

            std_eval = ClassificationEvaluator(schema=schema, window_size=1000)
            change_eval = StreamingChangeEvaluator(num_classes=NUM_CLASSES)

            print(f"\nInitial training for {model_name}...")
            for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc=f"Initial Train ({model_name})", leave=False):
                y_true = int(row[LABEL_NAME])
                X = np.array([row[c] for c in feature_cols], dtype=float)
                instance = LabeledInstance.from_array(schema, x=X, y_index=y_true)
                model.train(instance)
            print("Initial training completed.")

            progress = tqdm(enumerate(stream_ts), total=len(stream_ts),
                            desc=f"Prequential ({model_name})", leave=False)

            for step_idx, ts in progress:

                df_month = df_stream[df_stream["timestamp"] == ts]
                if df_month.empty:
                    continue

                print(f"\nProcessing month {ts} ({len(df_month)} instances)")

                y_true_list = []
                y_pred_list = []

                change_true_list = []
                change_pred_list = []
                sc_true_list = []
                sc_pred_list = []
                
                drifts = 0

                # 1) TEST su tutte le istanze del mese (senza training)
                for _, row in df_month.iterrows():
                    y_true = int(row[LABEL_NAME])
                    X = np.array([row[c] for c in feature_cols], dtype=float)
                    instance = LabeledInstance.from_array(schema, x=X, y_index=y_true)

                    y_pred = int(model.predict(instance))

                    y_true_list.append(y_true)
                    y_pred_list.append(y_pred)

                    # 4) Drift detection immediato
                    if drift_detector.detected_change():
                        drifts += 1
                        #print(f"⚠ Drift detected at timestamp {ts}, instance {row[PATCH_ID_COLUMN_NAME]}")

                    std_eval.update(y_true, y_pred)

                    last_state = getattr(change_eval, "_last_state", {})
                    if row[PATCH_ID_COLUMN_NAME] in last_state:
                        y_prev, y_pred_prev = last_state[row[PATCH_ID_COLUMN_NAME]]

                        gt_change = 1 if y_true != y_prev else 0
                        pred_change = 1 if y_pred != y_pred_prev else 0

                        change_true_list.append(gt_change)
                        change_pred_list.append(pred_change)

                        if gt_change == 1:
                            sc_true_list.append(y_true)
                            sc_pred_list.append(y_pred)

                    change_eval.update(row[PATCH_ID_COLUMN_NAME], y_true, y_pred)
                
                print(f"Month {ts} testing completed with {drifts} detected drifts.")

                # 4) Adattamento (se richiesto) su tutte le istanze del mese
                if ADAPT_ON_STREAM:
                    for _, row in df_month.iterrows():
                        y_true = int(row[LABEL_NAME])
                        X = np.array([row[c] for c in feature_cols], dtype=float)
                        instance = LabeledInstance.from_array(schema, x=X, y_index=y_true)
                        model.train(instance)
                        

                # 5) Log delle metriche
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
                    "detected_drifts": drifts
                }

                if LOG_IMAGES:
                    if y_true_list:
                        fig_cm = plot_confusion_matrix_image(y_true_list, y_pred_list, CLASS_NAMES, title=f"Confusion Matrix (ts={ts})")
                        log_data["Classification CM (per-timestamp)"] = wandb.Image(fig_cm, caption=f"Classification CM ts={ts}")
                        plt.close(fig_cm)

                    if change_true_list:
                        fig_change = plot_confusion_matrix_image(change_true_list, change_pred_list, ["no_change", "change"], title=f"Change Confusion Matrix (ts={ts})")
                        log_data["Change CM (per-timestamp)"] = wandb.Image(fig_change, caption=f"Change Confusion Matrix ts={ts}")
                        plt.close(fig_change)

                    if sc_true_list:
                        fig_sc = plot_confusion_matrix_image(sc_true_list, sc_pred_list, CLASS_NAMES, title=f"Semantic Change CM (ts={ts})")
                        log_data["Semantic Change CM (per-timestamp)"] = wandb.Image(fig_sc, caption=f"Semantic Change CM ts={ts}")
                        plt.close(fig_sc)

                wandb.log(log_data, step=ts)

                progress.set_postfix({
                    "acc": f"{std_eval.accuracy():.3f}",
                    "miou": f"{metrics.get('miou', 0.0):.3f}"
                })

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
            import traceback
            traceback.print_exc()

        finally:
            run.finish()

    return results

all_results_in_memory = []
all_files = [file for file in sorted(os.listdir(PROCESSED_DIR)) if file.endswith(".csv")]

OUTPUT_CSV_FILE = "drift_results.csv"
print(f"Saving incremental results to {OUTPUT_CSV_FILE}")

for file in tqdm(all_files, desc="Processing Embedding Files"):
    file_path = os.path.join(PROCESSED_DIR, file)
#file_path = "/Users/chiaranguyen/Desktop/SitsSCD/stream/embeddings/embeddings_dino_sat493m.csv"

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