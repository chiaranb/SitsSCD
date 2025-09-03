import pytorch_lightning as L
from torch.utils.data import DataLoader, Subset
from avalanche.benchmarks import benchmark_from_datasets
from avalanche.benchmarks.utils import make_avalanche_dataset
import torch
import time

class ImageDataModule(L.LightningDataModule):
    def __init__(        
        self,
        train_dataset,
        val_dataset_out,
        val_dataset_temporal,
        val_dataset_in,
        test_dataset_out,
        test_dataset_temporal,
        test_dataset_in,
        global_batch_size,
        num_workers,
        domain_shift_type,
        num_nodes=1,
        num_devices=1
    ):
        super().__init__()
        self._builders = {
            "train": train_dataset,
            "val_out": val_dataset_out,
            "val_temporal": val_dataset_temporal,
            "val_in": val_dataset_in,
            "test_out": test_dataset_out,
            "test_temporal": test_dataset_temporal,
            "test_in": test_dataset_in,
        }
        self.num_workers = num_workers
        self.batch_size = global_batch_size // (num_nodes * num_devices)
        self.domain_shift_type = domain_shift_type
        print(f"Each GPU will receive {self.batch_size} images")

    def setup(self, stage=None):
        start_time = time.time()

        # --- Train stream ---
        self.train_dataset = self._builders["train"]()
        if self.streaming:
            print("Setting up monthly streaming...")
            # Creazione di stream mensili
            train_streams = []
            #print(len(self.train_dataset.sits_ids)) #20
            for sits_number in range(len(self.train_dataset.sits_ids)):
                for month in self.train_dataset.month_list[sits_number]:
                    patches = self.train_dataset.get_patches_for_month(sits_number, month)
                    train_streams.append(make_avalanche_dataset(patches))
            self.train_streams = train_streams
            print(f"Total training experiences (months): {len(self.train_streams)}")
            #print(train_streams)
        else:
            # train normale senza streaming
            self.train_streams = [make_avalanche_dataset(self.train_dataset)]

        # --- Validation stream ---
        self.val_dataset_in = self._builders["val_in"]()
        val_streams = []
        print(len(self.val_dataset_in.sits_ids))
        for sits_number in range(len(self.val_dataset_in.sits_ids)):
            for month in self.val_dataset_in.month_list[sits_number]:
                patches = self.val_dataset_in.get_patches_for_month(sits_number, month)
                val_streams.append(make_avalanche_dataset(patches))
        self.val_streams = val_streams
        print(f"Total validation experiences (months): {len(self.val_streams)}")

        # --- Benchmark Avalanche ---
        self.benchmark = benchmark_from_datasets(
            train_stream=self.train_streams,
            test_stream=self.val_streams
        )

        end_time = time.time()
        print(f"Setup took {(end_time - start_time):.2f} seconds")

    def train_dataloader(self):
        # restituisce tutti i loader degli experience
        return [DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                           num_workers=self.num_workers, collate_fn=self.train_dataset.collate_fn)
                for ds in self.train_streams]

    def val_dataloader(self):
        return [DataLoader(ds, batch_size=1, shuffle=False,
                           num_workers=self.num_workers, collate_fn=self.val_dataset_in.collate_fn)
                for ds in self.val_streams]

    def test_dataloader(self):
        return [DataLoader(ds, batch_size=1, shuffle=False,
                           num_workers=self.num_workers, collate_fn=self.val_dataset_in.collate_fn)
                for ds in self.val_streams]