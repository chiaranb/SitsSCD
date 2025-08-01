import os
from models.module import SitsScdModel
import hydra
import wandb
from os.path import isfile, join
from shutil import copyfile

from omegaconf import OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.callbacks import LearningRateMonitor
from lightning_fabric.utilities.rank_zero import _get_rank


# Registering the "eval" resolver allows for advanced config
# interpolation with arithmetic operations in hydra:
# https://omegaconf.readthedocs.io/en/2.3_branch/how_to_guides.html
OmegaConf.register_new_resolver("eval", eval)

"""Initializes W&B with a unique ID, either from the config or generated."""
def wandb_init(cfg):
    directory = cfg.checkpoints.dirpath
    if isfile(join(directory, "wandb_id.txt")):
        with open(join(directory, "wandb_id.txt"), "r") as f:
            wandb_id = f.readline()
    else:
        rank = _get_rank()
        wandb_id = wandb.util.generate_id() # Generate a new W&B ID
        print(f"Generated wandb id: {wandb_id}")
        if rank == 0 or rank is None:
            with open(join(directory, "wandb_id.txt"), "w") as f:
                f.write(str(wandb_id))

    return wandb_id 

    
"""def wandb_init(cfg):
    if cfg.get("wandb") and cfg.wandb.id is not None:
        wandb_id = cfg.wandb.id
        print(f"[W&B] Using wandb ID from CLI/config: {wandb_id}")
    else:
        wandb_id = wandb.util.generate_id()
        print(f"[W&B] Generated new wandb ID: {wandb_id}")
    return wandb_id """

"""Loads the model from a checkpoint if available, otherwise initializes a new model."""
def load_model(cfg, dict_config, wandb_id, callbacks):
    directory = cfg.checkpoints.dirpath
    if isfile(join(directory, "last.ckpt")):
        checkpoint_path = join(directory, "last.ckpt")
        logger = instantiate(cfg.logger, id=wandb_id, resume="allow")
        model = SitsScdModel.load_from_checkpoint(checkpoint_path, cfg=cfg.model)
        ckpt_path = join(directory, "last.ckpt")
        print(f"Loading form checkpoint ... {ckpt_path}")
    else:
        ckpt_path = None
        logger = instantiate(cfg.logger, id=wandb_id, resume="allow")
        log_dict = {"model": dict_config["model"], "dataset": dict_config["dataset"]}
        logger._wandb_init.update({"config": log_dict})
        model = SitsScdModel(cfg.model)

    trainer, strategy = cfg.trainer, cfg.trainer.strategy
    trainer = instantiate(
        trainer, strategy=strategy, logger=logger, callbacks=callbacks,
    )
    return trainer, model, ckpt_path

"""def load_model(cfg, dict_config, wandb_id, callbacks):
    logger = instantiate(cfg.logger, id=wandb_id, resume="allow")
    
    # checkpoint from path in eval mode
    if cfg.mode == "eval" and cfg.eval.checkpoint_path is not None:
        checkpoint_path = cfg.eval.checkpoint_path
        print(f"[Eval] Loading checkpoint from: {checkpoint_path}")
        model = SitsScdModel.load_from_checkpoint(checkpoint_path, cfg=cfg.model)
    elif cfg.mode == "train" and cfg.train.checkpoint_path is not None:
        # checkpoint from path in train mode
        checkpoint_path = cfg.train.checkpoint_path
        print(f"[Train] Loading checkpoint from: {checkpoint_path}")
        model = SitsScdModel.load_from_checkpoint(checkpoint_path, cfg=cfg.model)
    else:
        print("No checkpoint found, initializing new model.")
        model = SitsScdModel(cfg.model)
        checkpoint_path = None

    trainer = instantiate(
        cfg.trainer,
        strategy=cfg.trainer.strategy,
        logger=logger,
        callbacks=callbacks,
    )

    return trainer, model, checkpoint_path """

"""Sets up the project directory and saves config for reproducibility."""
def project_init(cfg):
    print("Working directory set to {}".format(os.getcwd()))
    directory = cfg.checkpoints.dirpath
    os.makedirs(directory, exist_ok=True)
    copyfile(".hydra/config.yaml", join(directory, "config.yaml"))

"""Creates two checkpoint callbacks, a progress bar, and a learning rate monitor."""
def callback_init(cfg):
    monitor = cfg.checkpoints["monitor"]
    filename = cfg.checkpoints["filename"]
    cfg.checkpoints["monitor"] = monitor + "_out"
    cfg.checkpoints["filename"] = filename + "_out"
    checkpoint_callback_out = instantiate(cfg.checkpoints)
    cfg.checkpoints["monitor"] = monitor + "_in"
    cfg.checkpoints["filename"] = filename + "_in"
    checkpoint_callback_in = instantiate(cfg.checkpoints)
    progress_bar = instantiate(cfg.progress_bar)
    lr_monitor = LearningRateMonitor()
    callbacks = [checkpoint_callback_out, checkpoint_callback_in, progress_bar, lr_monitor]
    return callbacks

"""Instantiates the data module from the configuration."""
def init_datamodule(cfg):
    datamodule = instantiate(cfg.datamodule)
    return datamodule

"""Combines all initialization steps into a single function."""
def hydra_boilerplate(cfg):
    dict_config = OmegaConf.to_container(cfg, resolve=True)
    callbacks = callback_init(cfg)
    datamodule = init_datamodule(cfg)
    project_init(cfg)
    wandb_id = wandb_init(cfg)
    trainer, model, ckpt_path = load_model(cfg, dict_config, wandb_id, callbacks)
    return trainer, model, datamodule, ckpt_path


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    trainer, model, datamodule, ckpt_path = hydra_boilerplate(cfg)
    model.datamodule = datamodule
    if cfg.mode == "train":
        print(f"Using precision: {trainer.precision_plugin.precision}")
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    elif cfg.mode == "eval":
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
