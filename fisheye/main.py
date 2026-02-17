import platform

import hydra
import torch.multiprocessing as mp
from fisheye.runner import run_job
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    return run_job(cfg)


if __name__ == "__main__":
    # TODO (MVH): This check is a bit rough since we don't have the config yet, but 'spawn' is generally safer for
    #  Windows + multiprocessing anyways
    if platform.system() == "Windows":
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
    main()
