from fisheye.dataloaders.aris import ARISBatchedDataset
from fisheye.dataloaders.yolo import YOLOARISBatchedDataset


class DataloaderRegistry:
    """Dynamically load the appropriate dataloader based on the model."""

    _registry = {}

    @classmethod
    def register(cls, model_name: str, dataloader_cls):
        cls._registry[model_name] = dataloader_cls

    @classmethod
    def get_dataloader(cls, model_name: str, *args, **kwargs):
        if model_name not in cls._registry:
            raise ValueError(f"No dataloader found for model: {model_name}")
        return cls._registry[model_name]


# Manually register when creating new dataloaders
DataloaderRegistry.register("aris", ARISBatchedDataset)
DataloaderRegistry.register("yolo", YOLOARISBatchedDataset)


_DATASET_REGISTRY = {}


def register_dataset(config_type):
    def wrapper(dataset_cls):
        _DATASET_REGISTRY[config_type] = dataset_cls
        return dataset_cls

    return wrapper


def get_dataset_for_config(config):
    config_type = type(config)
    if config_type not in _DATASET_REGISTRY:
        raise ValueError(f"No dataset registered for config type: {config_type}")
    dataset_cls = _DATASET_REGISTRY[config_type]
    return dataset_cls(config)
