import wandb
from capymoa.classifier import (
    HoeffdingTree, NaiveBayes, SGDClassifier, KNN, EFDT, WeightedkNN,
    HoeffdingAdaptiveTree, LeveragingBagging, OnlineAdwinBagging, StreamingGradientBoostedTrees, AdaptiveRandomForestClassifier,
    DynamicWeightedMajority, OnlineBagging, OzaBoost, OnlineSmoothBoost, StreamingRandomPatches, SAMkNN, CSMOTE
)
from capymoa.evaluation import ClassificationEvaluator
from tqdm import tqdm
import pandas as pd
import numpy as np
from capymoa.instance import LabeledInstance
from capymoa.stream import Schema 
import os
from metrics import StreamingChangeEvaluator, NUM_CLASSES, CLASS_NAMES

# ---------------- Configuration ----------------
wandb.login()
PROJECT_NAME = "capymoa-streaming"

PROCESSED_DIR = "/Users/chiaranguyen/Desktop/SitsSCD/stream/emb_DINO"
PATCH_ID_COLUMN_NAME = "patch_id"
LABEL_NAME = "label"
OTHER_FEATURES = ["sits_id", "timestamp"]
MONTHS_PER_YEAR = 12
RANDOM_SEED = 42

# ---------------- Define Experiment Configurations ----------------
# --- MODIFIED: Replaced MODELS with a list of experiment configs ---
# Here you can define all the hyperparameter combinations you want to test.
# 'name' will be used for logging.
# 'model_class' is the classifier.
# 'params' is a dictionary of hyperparameters to pass to the classifier.
EXPERIMENT_CONFIGS = [
    {
        "name": "OnlineBagging_size_10",
        "model_class": OnlineBagging,
        "params": {"ensemble_size": 10, "random_seed": RANDOM_SEED}
    },
    {
        "name": "OnlineBagging_size_30",
        "model_class": OnlineBagging,
        "params": {"ensemble_size": 30, "random_seed": RANDOM_SEED}
    },
    {
        "name": "OnlineAdwinBagging_size_10",
        "model_class": OnlineAdwinBagging,
        "params": {"ensemble_size": 10, "random_seed": RANDOM_SEED}
    },
    {
        "name": "OnlineAdwinBagging_size_30",
        "model_class": OnlineAdwinBagging,
        "params": {"ensemble_size": 30, "random_seed": RANDOM_SEED}
    },
    {
        "name": "LeveragingBagging_size_10",
        "model_class": LeveragingBagging,
        "params": {"ensemble_size": 10, "random_seed": RANDOM_SEED}
    },
    {
        "name": "LeveragingBagging_size_30",
        "model_class": LeveragingBagging,
        "params": {"ensemble_size": 30, "random_seed": RANDOM_SEED}
    },
    {
        "name": "AdaptiveRandomForestClassifier_size_10",
        "model_class": AdaptiveRandomForestClassifier,
        "params": {"ensemble_size": 10, "random_seed": RANDOM_SEED}
    },
    {
        "name": "AdaptiveRandomForestClassifier_size_30",
        "model_class": AdaptiveRandomForestClassifier,
        "params": {"ensemble_size": 30, "random_seed": RANDOM_SEED}
    }
]

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

    feature_cols = [c for c in df.columns if c not in [PATCH_ID_COLUMN_NAME, LABEL_NAME, *OTHER_FEATURES]]
    schema = Schema.from_custom(
        feature_names=feature_cols,
        target_attribute_name=LABEL_NAME,
        values_for_class_label=list(range(len(CLASS_NAMES)))
    )

    run_name_base = os.path.basename(csv_path).replace(".csv", "")
    
    results = []

    # --- MODIFIED: Loop over the new config list ---
    for config in tqdm(EXPERIMENT_CONFIGS, desc=f"Models ({run_name_base})", leave=False):
        
        # Unpack the config
        model_name = config["name"]
        model_class = config["model_class"]
        model_params = config["params"]
        
        run = wandb.init(
            project=PROJECT_NAME,
            name=f"{run_name_base}_{model_name}", 
            
            # --- MODIFIED: Log all hyperparameters to wandb config ---
            config={
                "embedding_file": csv_path, 
                "model_name": model_name,
                "model_base": model_class.__name__,
                **model_params  # This unpacks the 'params' dict into the config
            },
            reinit=True
        )

        try:
            # --- MODIFIED: Instantiate model with its parameters ---
            model = model_class(schema=schema, **model_params)
            
            std_eval = ClassificationEvaluator(schema=schema, window_size=1000)
            change_eval = StreamingChangeEvaluator(num_classes=NUM_CLASSES)

            # 1️⃣ Train iniziale (2018)
            for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc=f"Initial Train ({model_name})", leave=False):
                y_true = int(row[LABEL_NAME])
                X = np.array([row[c] for c in feature_cols], dtype=float)
                instance = LabeledInstance.from_array(schema, x=X, y_index=y_true)
                model.train(instance)

            # 2️⃣ Prequential test (2019 mese per mese)
            pbar = tqdm(stream_ts, desc=f"Prequential ({model_name})", leave=False)
            for ts in pbar:
                df_month = df_stream[df_stream["timestamp"] == ts]
                if df_month.empty:
                    continue

                for _, row in df_month.iterrows():
                    y_true = int(row[LABEL_NAME])
                    X = np.array([row[c] for c in feature_cols], dtype=float)
                    instance = LabeledInstance.from_array(schema, x=X, y_index=y_true)

                    y_pred = int(model.predict(instance))
                    std_eval.update(y_true, y_pred)
                    change_eval.update(row[PATCH_ID_COLUMN_NAME], y_true, y_pred)
                    model.train(instance)

                # --- Log metrics ---
                metrics = change_eval.compute()
                log_data = {f"{k}": v for k, v in metrics.items()} 
                log_data[f"std_accuracy"] = std_eval.accuracy()
                log_data[f"std_precision"] = std_eval.precision()
                log_data[f"std_recall"] = std_eval.recall()
                log_data[f"std_f1"] = std_eval.f1_score()
                log_data[f"std_kappa"] = std_eval.kappa()
                
                wandb.log(log_data, step=ts) 

                pbar.set_postfix({
                    "acc": f"{std_eval.accuracy():.3f}",
                    "scs": f"{metrics.get('scs', 0.0):.3f}",
                    "miou": f"{metrics.get('miou', 0.0):.3f}"
                })

            # 3️⃣ Final metrics
            final_metrics = {
                "embedding": run_name_base,
                "model": model_name,
                "accuracy": std_eval.accuracy(),
                "precision": std_eval.precision(),
                "recall": std_eval.recall(),
                "f1": std_eval.f1_score(),
                "kappa": std_eval.kappa(),
                **change_eval.compute(),
            }
            results.append(final_metrics)

        except Exception as e:
            print(f"🚨 ERROR running model {model_name} on {run_name_base}: {e}")
            print("Skipping to next model...")
        
        finally:
            run.finish()

    return results

# ---------------- Master loop over all embeddings ----------------
# (This section is unchanged and correct)
all_results_in_memory = []
all_files = [file for file in sorted(os.listdir(PROCESSED_DIR)) if file.endswith(".csv")]

OUTPUT_CSV_FILE = "search_results_all_embeddings.csv"
print(f"Saving incremental results to {OUTPUT_CSV_FILE}")

for file in tqdm(all_files, desc="Processing Embedding Files"):
    file_path = os.path.join(PROCESSED_DIR, file)
    
    res = run_prequential_experiment(file_path)
    
    if res:
        df_batch = pd.DataFrame(res)
        # Check if file exists to determine if we need to write the header
        # This is important now that new param columns might be added
        write_header = not os.path.exists(OUTPUT_CSV_FILE)
        
        df_batch.to_csv(
            OUTPUT_CSV_FILE, 
            mode='a',
            header=write_header, 
            index=False
        )
        all_results_in_memory.extend(res)

# ---------------- Save combined results ----------------
# (This section is unchanged and correct)
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