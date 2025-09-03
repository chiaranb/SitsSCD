import os
import hydra
import torch
from avalanche.training.supervised import Naive
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np

from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np


# Import your dataset and model
from data.data import Muds
from models.networks.multiutae import MultiUTAE


class DictDatasetWrapper:
    """Wrapper that maintains the dictionary format for your model"""
    
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        # Get the original sample (should be a dict)
        sample = self.original_dataset[idx]
        
        # Extract data and target
        x = sample["data"]   # input images - shape [3, 128, 128] or similar
        y = sample["gt"]     # segmentation mask - shape [128, 128] or similar
        
        # Ensure consistent tensor types
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
            
        # For segmentation tasks, ensure y is the right type
        if y.dtype != torch.long:
            y = y.long()
        
        # Return as tuple (x, y) for Avalanche compatibility
        return x.float(), y.long()


class ModelWrapper(torch.nn.Module):
    """Wrapper to adapt your model to work with Avalanche's tensor input format"""
    
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        
    def forward(self, x):
        # Your model expects batch["data"], so create the batch dict
        batch = {"data": x}
        
        # Call your original model with the batch format it expects
        # Note: if your model returns multiple outputs, handle them appropriately
        output = self.original_model(batch)
        
        # If your model returns a tuple/dict, extract the main output
        if isinstance(output, dict):
            # If output is a dict, return the main prediction
            if "pred" in output:
                return output["pred"]
            elif "output" in output:
                return output["output"]
            else:
                # Return the first value in the dict
                return list(output.values())[0]
        elif isinstance(output, tuple):
            # If output is a tuple, return the first element (usually predictions)
            return output[0]
        else:
            # Direct tensor output
            return output


class StreamingDatasetWrapper:
    """Wrapper to create streaming chunks from your dataset"""
    
    def __init__(self, dataset, chunk_size=100):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.total_samples = len(dataset)
        
    def get_chunks(self):
        """Generator that yields chunks of the dataset"""
        for i in range(0, self.total_samples, self.chunk_size):
            end_idx = min(i + self.chunk_size, self.total_samples)
            indices = list(range(i, end_idx))
            chunk = Subset(self.dataset, indices)
            yield chunk


def create_avalanche_dataset(dataset_chunk, task_id=0):
    """Convert a dataset chunk to AvalancheDataset format"""
    
    # Wrap the chunk to ensure proper (x, y) tuple format
    wrapped_chunk = DictDatasetWrapper(dataset_chunk)
    
    # Create AvalancheDataset directly from the wrapped chunk
    avl_dataset = AvalancheDataset(wrapped_chunk)
    return avl_dataset


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    # --------------------
    # 1. Dataset streaming setup
    # --------------------
    train_set = Muds(
        path="/teamspace/studios/this_studio/SitsSCD/datasets/Muds", 
        split="train", 
        domain_shift_type="none", 
        img_size=128, 
        true_size=1024
    )
    test_set = Muds(
        path="/teamspace/studios/this_studio/SitsSCD/datasets/Muds", 
        split="test", 
        domain_shift_type="none", 
        img_size=128, 
        true_size=1024
    )

    # Create streaming wrapper
    streaming_wrapper = StreamingDatasetWrapper(train_set, chunk_size=64)
    
    # Create train datasets for streaming (multiple chunks)
    train_datasets = []
    for i, chunk in enumerate(streaming_wrapper.get_chunks()):
        avl_chunk = create_avalanche_dataset(chunk, task_id=i)
        train_datasets.append(avl_chunk)
    
    # Create test dataset
    test_avl = create_avalanche_dataset(test_set, task_id=0)
    test_datasets = [test_avl]  # Same test set for all tasks
    
    # Create benchmark
    benchmark = benchmark_from_datasets(
        train=train_datasets,
        test=test_datasets
    )

    print(f"Created benchmark with {len(train_datasets)} streaming experiences")

    # --------------------
    # 2. Model setup
    # --------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create your original model
    original_model = MultiUTAE(
        input_dim=3,
        num_classes=2,
        in_features=512,
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        agg_mode="att_group",
        encoder_norm="group",
        n_head=16,
        d_k=4,
        pad_value=0,
        padding_mode="reflect",
        T=730,
        offset=0
    )
    
    # Wrap the model to handle tensor inputs from Avalanche
    model = ModelWrapper(original_model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # For segmentation tasks, you might want to use different loss
    # criterion = torch.nn.CrossEntropyLoss()  # For classification
    # For segmentation, consider:
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)  # Ignore invalid pixels

    # --------------------
    # 3. Logging & metrics
    # --------------------
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger],
    )

    # --------------------
    # 4. Streaming strategy
    # --------------------
    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=1,      # mini-batch size
        train_epochs=5,       # epochs per experience
        eval_mb_size=8,       # evaluation batch size
        device=device,
        evaluator=eval_plugin,
    )

    # --------------------
    # 5. Streaming training loop
    # --------------------
    print("Starting streaming continual learning...")
    
    for experience_id, experience in enumerate(benchmark.train_stream):
        print(f"\n--- Experience {experience_id} ---")
        print(f"Training on {len(experience.dataset)} samples")
        
        # Train on current experience
        strategy.train(experience)
        
        # Evaluate on test set
        print("Evaluating...")
        results = strategy.eval(benchmark.test_stream)
        
        # Optional: Print some metrics
        if hasattr(results, 'keys'):
            for key, value in results.items():
                if 'Top1_Acc' in key:
                    print(f"{key}: {value:.4f}")
    
    print("\nStreaming training completed!")


if __name__ == "__main__":
    main()