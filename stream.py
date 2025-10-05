from capymoa.stream import CSVStream
from capymoa.classifier import HoeffdingTree, NaiveBayes, SGDClassifier, KNN
from capymoa.evaluation import ClassificationEvaluator
import math

CSV_PATH = "pixelwise_embeddings_test_2.csv"

# Dizionario dei modelli da testare
MODELS = {
    "HoeffdingTree": HoeffdingTree,
    "NaiveBayes": NaiveBayes,
    "SGD": lambda schema: SGDClassifier(schema=schema, loss="log_loss"),
    "KNN": lambda schema: KNN(schema=schema, k=5, window_size=1000),
}

results = {}

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

    i = 0
    while stream.has_more_instances():
        instance = stream.next_instance()
        prediction = model.predict(instance)
        evaluator.update(instance.y_index, prediction)
        model.train(instance)

        i += 1
        if i % 1000 == 0:
            print(f"[{i}] Acc: {evaluator.accuracy():.4f}")

    # Salva risultati finali
    acc = evaluator.accuracy()
    kappa = evaluator.kappa()
    results[model_name] = {"accuracy": acc, "kappa": kappa}

    print(f"Final {model_name} -> Acc: {acc:.4f}, Kappa: {kappa:.4f}")

print("\n=== Summary ===")
for model_name, metrics in results.items():
    print(f"{model_name}: Acc={metrics['accuracy']:.4f}, Kappa={metrics['kappa']:.4f}")