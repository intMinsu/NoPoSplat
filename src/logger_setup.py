from omegaconf import OmegaConf
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only

class WandbLoggerManager():
    _logger = None

    @classmethod
    def setup(cls, cfg_dict, output_dir):
        if cls._logger is None:
            wandb_config = {
                "project": cfg_dict.wandb.project,
                "mode": cfg_dict.wandb.mode,
                "name": f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
                "tags": cfg_dict.wandb.get("tags", None),
                "log_model": False,
                "save_dir": output_dir,
                "config": OmegaConf.to_container(cfg_dict),
            }
            cls._logger = WandbLogger(**wandb_config)
        return cls._logger

    @classmethod
    def get_logger(cls):
        if cls._logger is None:
            raise RuntimeError("WandbLogger has not been initialized. Call `setup` first.")
        return cls._logger