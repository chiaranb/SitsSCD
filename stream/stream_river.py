import wandb
from capymoa.evaluation import ClassificationEvaluator
from tqdm import tqdm
import pandas as pd
import numpy as np
from capymoa.instance import LabeledInstance
from capymoa.stream import Schema 
import os
from metrics import StreamingChangeEvaluator, NUM_CLASSES, CLASS_NAMES

from river import linear_model, optim, preprocessing, feature_extraction, compose, multiclass

from river_wrapper import RiverClassifier


# ---------------- Configuration ----------------
#wandb.login()
PROJECT_NAME = "capymoa-streaming"

PROCESSED_DIR = "/Users/chiaranguyen/Desktop/SitsSCD/stream/emb_DINO"
PATCH_ID_COLUMN_NAME = "patch_id"
LABEL_NAME = "label"
OTHER_FEATURES = ["sits_id", "timestamp"]
MONTHS_PER_YEAR = 12

# ---------------- Define Pipeline Combinations ----------------
# This is where you define all the pipelines you want to test.
# The key is the name that will appear in WandB.
# The value is a function that *returns* an instantiated River pipeline.

# Set parameters for projections
N_COMPONENTS = 256
RANDOM_SEED = 42

PIPELINE_DEFINITIONS = {
    "Perceptron_OneVsRest": lambda: compose.Pipeline(
        multiclass.OneVsRestClassifier(linear_model.Perceptron())
    ),
    "Logistic_OneVsRest": lambda: compose.Pipeline(
        multiclass.OneVsRestClassifier(linear_model.LogisticRegression())
    ),
    "StdScale_Logistic_OneVsRest": lambda: compose.Pipeline(
        preprocessing.StandardScaler(),
        multiclass.OneVsRestClassifier(linear_model.LogisticRegression())
    ),
    "RobustScale_Logistic_OneVsRest": lambda: compose.Pipeline(
        preprocessing.RobustScaler(),
        multiclass.OneVsRestClassifier(linear_model.LogisticRegression())
    ),
    "StdScale_Perceptron_OneVsRest": lambda: compose.Pipeline(
        preprocessing.StandardScaler(),
        multiclass.OneVsRestClassifier(linear_model.Perceptron())
    ),
    "RobustScale_Perceptron_OneVsRest": lambda: compose.Pipeline(
        preprocessing.RobustScaler(),
        multiclass.OneVsRestClassifier(linear_model.Perceptron())
    ),
}


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

    # --- 3. Create the final models to run from the definitions ---
    # This dictionary will hold the final, callable lambdas for the wrapper
    MODELS_TO_RUN = {}
    for pipe_name, pipe_fn in PIPELINE_DEFINITIONS.items():
        # This lambda captures the schema (s) and pipeline function (p)
        # It creates a RiverClassifier with an *instantiated* pipeline
        MODELS_TO_RUN[pipe_name] = (lambda s=schema, p=pipe_fn: 
            RiverClassifier(
                schema=s,
                river_model_instance=p() # p() calls the lambda, e.g., compose.Pipeline(...)
            )
        )

    # --- 4. Loop through the dynamically created pipelines ---
    for model_name, model_lambda in tqdm(MODELS_TO_RUN.items(), desc=f"Pipelines ({run_name_base})", leave=False):
        
        run = wandb.init(
            project=PROJECT_NAME,
            # Name will be e.g., "my_embedding_StdScale_GRP_LR"
            name=f"{run_name_base}_{model_name}", 
            config={
                "embedding_file": csv_path, 
                "pipeline": model_name
            },
            reinit=True 
        )

        try:
            # Call the lambda to get the fully wrapped model
            model = model_lambda() 
            
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
                "model": model_name, # This will now be the pipeline name
                "accuracy": std_eval.accuracy(),
                "precision": std_eval.precision(),
                "recall": std_eval.recall(),
                "f1": std_eval.f1_score(),
                "kappa": std_eval.kappa(),
                **change_eval.compute(),
            }
            results.append(final_metrics)

        except Exception as e:
            print(f"🚨 ERROR running pipeline {model_name} on {run_name_base}: {e}")
            print("Skipping to next pipeline...")
        
        finally:
            run.finish()

    return results

# ---------------- Master loop over all embeddings ----------------
# (This section is unchanged and correct)
all_results_in_memory = []
all_files = [file for file in sorted(os.listdir(PROCESSED_DIR)) if file.endswith(".csv")]

OUTPUT_CSV_FILE = "search_results_all_embeddings.csv"
print(f"Saving incremental results to {OUTPUT_CSV_FILE}")

#for file in tqdm(all_files, desc="Processing Embedding Files"):
#file_path = os.path.join(PROCESSED_DIR, file)
file_path = "/Users/chiaranguyen/Desktop/SitsSCD/stream/emb_DINO/embeddings_dino_small.csv"

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

print("\n✅ All embedding evaluations completed.")