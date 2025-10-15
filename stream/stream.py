import wandb
from capymoa.stream import CSVStream
from capymoa.classifier import (
    HoeffdingTree, NaiveBayes, SGDClassifier, KNN,
    HoeffdingAdaptiveTree, LeveragingBagging, MajorityClass
)
from capymoa.evaluation import ClassificationEvaluator
from tqdm import tqdm
import pandas as pd

# === Configurazione wandb ===
wandb.login()  # esegui una volta da terminale: wandb login
PROJECT_NAME = "capymoa-streaming"

CSV_PATH = "embeddings_sample.csv"

# Dizionario dei modelli da testare
MODELS = {
    "KNN": lambda schema: KNN(schema=schema, k=5, window_size=2),
    "SGD": lambda schema: SGDClassifier(schema=schema, loss="log_loss"),
    "HoeffdingTree": HoeffdingTree,
    "NaiveBayes": NaiveBayes,
    "HoeffdingAdaptiveTree": HoeffdingAdaptiveTree,
    "LeveragingBagging": LeveragingBagging,
    "MajorityClass": MajorityClass,
}

# Conta totale delle istanze
n_instances = sum(1 for _ in open(CSV_PATH)) - 1

# Avvio unica run wandb
run = wandb.init(
    project=PROJECT_NAME,
    name="all_models_streaming",
    config={
        "dataset": CSV_PATH,
        "n_instances": n_instances,
        "models": list(MODELS.keys())
    }
)

results = {}

for model_name, model_class in MODELS.items():
    print(f"\n=== Training {model_name} ===")

    # Ricarica stream per ogni modello
    stream = CSVStream(
        CSV_PATH,
        target_attribute_name="label",
        values_for_class_label=[0, 1, 2, 3, 4, 5],
        class_index=3
    )
    schema = stream.get_schema()

    # Istanzia modello ed evaluator
    model = model_class(schema) if callable(model_class) else model_class(schema)
    evaluator = ClassificationEvaluator(schema=schema, window_size=5000)

    pbar = tqdm(range(n_instances), desc=f"{model_name}", ncols=100)

    for i in pbar:
        if not stream.has_more_instances():
            break
        instance = stream.next_instance()
        prediction = model.predict(instance)
        evaluator.update(instance.y_index, prediction)
        model.train(instance)

        # Log progressivo ogni 5000 istanze
        if (i + 1) % 5000 == 0:
            acc = evaluator.accuracy()
            wandb.log({
                f"{model_name}/step": i + 1,
                f"{model_name}/accuracy": evaluator.accuracy(),
                f"{model_name}/precision": evaluator.precision(),
                f"{model_name}/recall": evaluator.recall(),
                f"{model_name}/f1": evaluator.f1_score(),
                f"{model_name}/kappa": evaluator.kappa(),
            })
            pbar.set_postfix({"acc": f"{acc:.4f}"})

    pbar.close()

    metrics = {
        "accuracy": evaluator.accuracy(),
        "kappa": evaluator.kappa(),
        "precision": evaluator.precision(),
        "recall": evaluator.recall(),
        "f1": evaluator.f1_score(),
    }
    results[model_name] = metrics
    print(f"Final {model_name} -> Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, Kappa: {metrics['kappa']:.4f}")

df = pd.DataFrame(results).T

df_reset = df.reset_index().rename(columns={"index": "model"})
df_reset = df_reset.sort_values(by="accuracy", ascending=False).reset_index(drop=True)
df_reset.to_csv("streaming_results.csv", index=False)
print("Risultati salvati in 'streaming_results.csv'")

# Log finale su wandb
wandb.log({"final_results": wandb.Table(dataframe=df_reset)})
wandb.finish()