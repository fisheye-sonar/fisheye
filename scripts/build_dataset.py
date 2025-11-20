import hydra
from omegaconf import DictConfig
from pathlib import Path

from fisheye.dataset.builder import DatasetBuilder
from fisheye.dataset.enums import DatasetFormat


@hydra.main(config_path="../configs", config_name="dataset", version_base="1.3")
def main(cfg: DictConfig):
    """Build dataset from ARIS files and XML annotations.

    Users can configure the dataset creation in two ways:

    1. Using the Hydra config file:
       Edit parameters in `configs/dataset.yaml` and run:
           $ python build_dataset.py

    2. Overriding parameters from the command line:
           $ python build_dataset.py aris_dir=/path/to/aris xml_dir=/path/to/xml out_dir=/path/to/output
    """
    builder = DatasetBuilder(
        aris_dir=Path(cfg.aris_dir),
        xml_dir=Path(cfg.xml_dir),
        out_dir=Path(cfg.out_dir),
        dataset_format=DatasetFormat(cfg.dataset_format),
        padding=cfg.padding,
        min_padding_px=cfg.min_padding_px,
    )
    builder()


if __name__ == "__main__":
    main()
