import wandb
from capymoa.evaluation import ClassificationEvaluator
from tqdm import tqdm
import pandas as pd
import numpy as np
from capymoa.instance import LabeledInstance
from capymoa.stream import Schema 
from capymoa.base import SKClassifier
from capymoa.type_alias import LabelIndex
from sklearn import linear_model, multiclass
from capymoa.stream.preprocessing import ClassifierPipeline, MOATransformer
from capymoa.drift.detectors import ADWIN
from moa.streams.filters import NormalisationFilter

import os
from metrics import StreamingChangeEvaluator, NUM_CLASSES, CLASS_NAMES

# ---------------- Configuration ----------------
wandb.login()
PROJECT_NAME = "capymoa-streaming"

ADAPT_ON_STREAM = True  # True for prequential, False for test-only

PROCESSED_DIR = "/Users/chiaranguyen/Desktop/SitsSCD/stream/embeddings"
PATCH_ID_COLUMN_NAME = "patch_id"
LABEL_NAME = "label"
OTHER_FEATURES = ["sits_id", "timestamp"]
MONTHS_PER_YEAR = 12
RANDOM_SEED = 42

# ---------------- Define Pipeline Combinations ----------------

PIPELINE_DEFINITIONS = {
    "SGDClassifier_OneVsRest": lambda schema: SKClassifier(
        schema=schema,
        sklearner=multiclass.OneVsRestClassifier(
            linear_model.SGDClassifier(random_state=RANDOM_SEED, loss="log_loss")
        )),
}


def label_equals_prediction(instance: LabeledInstance, prediction: LabelIndex) -> LabelIndex:
    label = instance.y_index
    return int(label == prediction)


# ---------------- Helper: prequential loop ----------------
def run_prequential_experiment(csv_path: str):
    print(f"\n=== Loading {csv_path} ===")
    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    unique_ts = sorted(df["timestamp"].unique())

    # Split temporale
    train_ts = unique_ts[:MONTHS_PER_YEAR]
    stream_ts = unique_ts[MONTHS_PER_YEAR:]

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
    results = []

    for model_name, model_fn in tqdm(PIPELINE_DEFINITIONS.items(), desc=f"Pipelines ({run_name_base})", leave=False):
        run_name = f"{run_name_base}_{model_name}"

        run = wandb.init(
            project=PROJECT_NAME,
            name=run_name,
            config={
                "embedding_file": csv_path,
                "pipeline": model_name,
                "adaptation": ADAPT_ON_STREAM
            },
            reinit=True
        )

        try:
            model = model_fn(schema)
            std_eval = ClassificationEvaluator(schema=schema)
            change_eval = StreamingChangeEvaluator(num_classes=NUM_CLASSES)

            print(f"\nInitial training for {model_name}...")

            for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc=f"Initial Train ({model_name})", leave=False):
                y_true = int(row[LABEL_NAME])
                X = np.array([row[c] for c in feature_cols], dtype=float)
                instance = LabeledInstance.from_array(schema, x=X, y_index=y_true)
                model.train(instance)

            print("Initial training completed.")

            pbar = tqdm(stream_ts, desc=f"Prequential ({model_name})", leave=False)

            for ts in pbar:
                df_month = df_stream[df_stream["timestamp"] == ts]
                if df_month.empty:
                    continue
                
                print(f"Testing timestamp: {ts} with {len(df_month)} instances")
                for _, row in df_month.iterrows():
                    y_true = int(row[LABEL_NAME])
                    X = np.array([row[c] for c in feature_cols], dtype=float)
                    instance = LabeledInstance.from_array(schema, x=X, y_index=y_true)

                    y_pred = int(model.predict(instance))
                    std_eval.update(y_true, y_pred)
                    change_eval.update(row[PATCH_ID_COLUMN_NAME], y_true, y_pred)

                if ADAPT_ON_STREAM:
                    print(f"Training on month {ts}...")
                    for _, row in df_month.iterrows():
                        y_true = int(row[LABEL_NAME])
                        X = np.array([row[c] for c in feature_cols], dtype=float)
                        instance = LabeledInstance.from_array(schema, x=X, y_index=y_true)
                        model.train(instance)
                    

                    wandb.log({
                        "std_accuracy": std_eval.accuracy(),
                        "std_precision": std_eval.precision(),
                        "std_recall": std_eval.recall(),
                        "std_f1": std_eval.f1_score(),
                        "std_kappa": std_eval.kappa(),
                        "std_kappa_m": std_eval.kappa_m(),
                        "std_kappa_t": std_eval.kappa_t(),
                    }, step=ts)

                metrics = change_eval.compute()
                log_data = {**metrics,
                            "std_accuracy": std_eval.accuracy(),
                            "std_precision": std_eval.precision(),
                            "std_recall": std_eval.recall(),
                            "std_f1": std_eval.f1_score(),
                            "std_kappa": std_eval.kappa(),
                            "std_kappa_m": std_eval.kappa_m(),
                            "std_kappa_t": std_eval.kappa_t()
                            }

                wandb.log(log_data, step=ts)

                pbar.set_postfix({
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
            print(f"ERROR running pipeline {model_name} on {run_name_base}: {e}")
            print("Skipping to next pipeline...")

        finally:
            run.finish()
    return results


# ---------------- Master loop over all embeddings ----------------
all_results_in_memory = []
all_files = [file for file in sorted(os.listdir(PROCESSED_DIR)) if file.endswith(".csv")]

OUTPUT_CSV_FILE = "search_results_all_embeddings_new.csv"
print(f"Saving incremental results to {OUTPUT_CSV_FILE}")

for file in tqdm(all_files, desc="Processing Embedding Files"):
    file_path = os.path.join(PROCESSED_DIR, file)
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