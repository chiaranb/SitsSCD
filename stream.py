from capymoa.stream import CSVStream
from capymoa.classifier import (
    HoeffdingTree, NaiveBayes, SGDClassifier, KNN,
    HoeffdingAdaptiveTree, LeveragingBagging, MajorityClass
)
from capymoa.evaluation import ClassificationEvaluator
from tqdm import tqdm
import pandas as pd

CSV_PATH = "embeddings_sorted.csv"

# Dizionario dei modelli da testare
MODELS = {
    "HoeffdingTree": HoeffdingTree,
    "NaiveBayes": NaiveBayes,
    "SGD": lambda schema: SGDClassifier(schema=schema, loss="log_loss"),
    "KNN": lambda schema: KNN(schema=schema, k=5, window_size=1000),
    "HoeffdingAdaptiveTree": HoeffdingAdaptiveTree,
    "LeveragingBagging": LeveragingBagging,
    "MajorityClass": MajorityClass,
}

results = {}
n_instances = sum(1 for _ in open(CSV_PATH)) - 1  # righe totali nel CSV

for model_name, model_class in MODELS.items():
    print(f"\n=== Training {model_name} ===")

    # Ricarica stream
    stream = CSVStream(
        CSV_PATH,
        target_attribute_name="label",
        values_for_class_label=[0, 1, 2, 3, 4, 5]
    )
    schema = stream.get_schema()

    # Istanzia modello
    model = model_class(schema) if callable(model_class) else model_class(schema)
    evaluator = ClassificationEvaluator(schema=schema, window_size=1000)

    # tqdm con aggiornamento dinamico
    pbar = tqdm(range(n_instances), desc=f"{model_name}", ncols=100)

    for i in pbar:
        if not stream.has_more_instances():
            break
        instance = stream.next_instance()
        prediction = model.predict(instance)
        evaluator.update(instance.y_index, prediction)
        model.train(instance)

        # Mostra accuracy ogni 1000 istanze
        if (i + 1) % 1000 == 0:
            acc = evaluator.accuracy()
            pbar.set_postfix({"acc": f"{acc:.4f}"})

    pbar.close()

    # Metriche finali
    metrics = {
        "accuracy": evaluator.accuracy(),
        "kappa": evaluator.kappa(),
        "precision": evaluator.precision(),
        "recall": evaluator.recall(),
        "f1": evaluator.f1(),
        "gmean": evaluator.gmean(),
        "balanced_acc": evaluator.balanced_accuracy(),
    }
    results[model_name] = metrics
    print(f"Final {model_name} -> Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, Kappa: {metrics['kappa']:.4f}")

# Salvataggio risultati
df = pd.DataFrame(results).T
df.to_csv("streaming_results.csv", index_label="model")
print("\n✅ Risultati salvati in 'streaming_results.csv'")