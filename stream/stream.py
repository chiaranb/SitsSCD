import wandb
from capymoa.stream import CSVStream
from capymoa.classifier import (
    HoeffdingTree, NaiveBayes, SGDClassifier, KNN,
    HoeffdingAdaptiveTree, LeveragingBagging, MajorityClass, OnlineAdwinBagging, StreamingGradientBoostedTrees, AdaptiveRandomForestClassifier
)
from capymoa.drift.detectors import ADWIN
from capymoa.evaluation import ClassificationEvaluator
from tqdm import tqdm
import pandas as pd
import numpy as np
from capymoa.instance import LabeledInstance
# --- 1. Importa Schema ---
from capymoa.stream import Schema 

# Importa la classe e le costanti dal tuo file locale
from metrics import StreamingChangeEvaluator, NUM_CLASSES, CLASS_NAMES

# === Configurazione wandb ===
wandb.login()
PROJECT_NAME = "capymoa-streaming"

CSV_PATH = "/Users/chiaranguyen/Desktop/SitsSCD/embeddings_2019.csv"
PATCH_ID_COLUMN_NAME = "patch_id" 
LABEL_NAME = "label"
OTHER_FEATURES = ["sits_id", "timestamp"] 
# -----------------------------------------------------------

MODELS = {
    "LeveragingBagging": lambda schema: LeveragingBagging(schema=schema, ensemble_size=30, minibatch_size=100),
    "AdaptiveRandomForest": lambda schema: AdaptiveRandomForestClassifier(schema=schema, ensemble_size=30, minibatch_size=100, drift_detection_method=ADWIN()),
    "OnlineAdwinBagging": lambda schema: OnlineAdwinBagging(schema=schema, ensemble_size=30, minibatch_size=100),
}

# --- Caricamento Dati ---
print(f"Loading data from {CSV_PATH}...")
df_stream = pd.read_csv(CSV_PATH)
n_instances = len(df_stream)
print(f"Loaded {n_instances} instances.")

# --- 2. Costruzione Schema Corretta ---
# Determina le colonne delle feature (tutto tranne ID e label)
feature_columns = [col for col in df_stream.columns if col not in [PATCH_ID_COLUMN_NAME, LABEL_NAME, *OTHER_FEATURES]]
# Usa la tua lista di nomi di classi definita in streaming_metrics.py
class_labels_str = [str(name) for name in CLASS_NAMES]
class_labels_values = list(range(len(CLASS_NAMES)))

# Costruisci lo schema manualmente
schema = Schema.from_custom(
    feature_names=feature_columns,
    target_attribute_name=LABEL_NAME,
    values_for_class_label=class_labels_values
)
# -------------------------------------

# Avvio unica run wandb
run = wandb.init(
    project=PROJECT_NAME,
    name="change_metrics_streaming_eval",
    config={
        "dataset": CSV_PATH,
        "n_instances": n_instances,
        "models": list(MODELS.keys())
    }
)

results = {}

for model_name, model_class in MODELS.items():
    print(f"\n=== Training {model_name} ===")

    # Istanzia modello ed evaluator personalizzato
    model = model_class(schema)
    std_evaluator = ClassificationEvaluator(schema=schema, window_size=1000)
    change_evaluator = StreamingChangeEvaluator(num_classes=NUM_CLASSES, ignore_index=None)

    pbar = tqdm(df_stream.iterrows(), total=n_instances, desc=f"{model_name}", ncols=100)

    for i, row in pbar:
        patch_id = getattr(row, PATCH_ID_COLUMN_NAME)
        y_true_index = int(getattr(row, LABEL_NAME))
        feature_values = np.array([getattr(row, c) for c in feature_columns], dtype=float)
        
        # Crea l'istanza (questo codice ora è corretto)
        instance = LabeledInstance.from_array(
            schema=schema,
            x=feature_values,
            y_index=y_true_index
        )
        
        # 3. Predici (questo non dovrebbe più crashare)
        prediction_array = model.predict(instance)
        y_pred_index = int(prediction_array)

        # 4. Valuta
        std_evaluator.update(y_true_index, y_pred_index)
        change_evaluator.update(patch_id, y_true_index, y_pred_index)

        # 5. Addestra
        model.train(instance)

        # 6. Log progressivo
        if (i + 1) % 5000 == 0:
            metrics = change_evaluator.compute() 
            log_data = {f"{model_name}/{k}": v for k, v in metrics.items()}
            log_data[f"{model_name}/std_accuracy"] = std_evaluator.accuracy()
            log_data[f"{model_name}/std_precision"] = std_evaluator.precision()
            log_data[f"{model_name}/std_recall"] = std_evaluator.recall()
            log_data[f"{model_name}/std_f1"] = std_evaluator.f1_score()
            log_data[f"{model_name}/step"] = i + 1
            wandb.log(log_data)
            pbar.set_postfix({
                "acc": f"{std_evaluator.accuracy():.2f}",
                "scs": f"{metrics['scs']:.2f}"
            })

    pbar.close()

    # Calcolo finale
    change_metrics = change_evaluator.compute()
    std_metrics = {
        "accuracy": std_evaluator.accuracy(),
        "precision": std_evaluator.precision(),
        "recall": std_evaluator.recall(),
        "f1": std_evaluator.f1_score(),
    }
    final_metrics = {**std_metrics, **change_metrics}
    results[model_name] = final_metrics
    print(f"Final {model_name} -> Acc: {final_metrics['accuracy']:.4f}, mIoU: {final_metrics['miou']:.4f}, SCS: {final_metrics['scs']:.4f}, BC: {final_metrics['bc']:.4f}, SC: {final_metrics['sc']:.4f}")

# --- Salvataggio e Log Finale ---
df = pd.DataFrame(results).T
df_reset = df.reset_index().rename(columns={"index": "model"})
df_reset = df_reset.sort_values(by="scs", ascending=False).reset_index(drop=True)
df_reset.to_csv("streaming_change_results.csv", index=False)
print("Risultati salvati in 'streaming_change_results.csv'")

wandb.log({"final_change_results": wandb.Table(dataframe=df_reset)})
wandb.finish()