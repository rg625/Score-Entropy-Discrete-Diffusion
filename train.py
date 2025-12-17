"""Training and evaluation"""

import hydra
import os
import sys
import numpy as np
import run_train
import utils
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf, open_dict

local_tmp = os.environ.get("SLURM_TMPDIR", "/tmp")

print(f"Setting TMPDIR to: {local_tmp}")

# Set up caching directories for compiled kernels and temporary files
os.environ["TRITON_CACHE_DIR"] = os.path.join(local_tmp, "triton_cache")
os.environ["TMPDIR"] = os.path.join(local_tmp, "sedd_tmp")

os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    ngpus = cfg.ngpus
    
    # Handle resuming from a previous run directory if specified
    if "load_dir" in cfg:
        hydra_cfg_path = os.path.join(cfg.load_dir, ".hydra/hydra.yaml")
        hydra_cfg = OmegaConf.load(hydra_cfg_path).hydra

        cfg = utils.load_hydra_config_from_run(cfg.load_dir)
        
        work_dir = cfg.work_dir
        utils.makedirs(work_dir)
    else:
        # Standard run setup
        hydra_cfg = HydraConfig.get()
        work_dir = hydra_cfg.run.dir if hydra_cfg.mode == RunMode.RUN else os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
        utils.makedirs(work_dir)

    # Inject runtime variables into the config
    with open_dict(cfg):
        cfg.ngpus = ngpus
        cfg.work_dir = work_dir
        # Removed cfg.wandb_name assignment as it is no longer used

    # Run the training pipeline
    port = int(np.random.randint(10000, 20000))
    logger = utils.get_logger(os.path.join(work_dir, "logs"))

    hydra_cfg = HydraConfig.get()
    if hydra_cfg.mode != RunMode.RUN:
        logger.info(f"Run id: {hydra_cfg.job.id}")

    try:
        # 'forkserver' is safer for CUDA + Multiprocessing than 'fork'
        mp.set_start_method("forkserver")
        mp.spawn(run_train.run_multiprocess, args=(ngpus, cfg, port), nprocs=ngpus, join=True)
    except Exception as e:
        logger.critical(e, exc_info=True)


if __name__ == "__main__":
    main()