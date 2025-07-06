
from typing import Callable

from torch.utils.data import Dataset
from nuclei_segmentation.datasets.pannuke import PanNukeDataset


def select_dataset(
    dataset_name: str, split: str, dataset_config: dict, transforms: Callable = None,
) -> Dataset:

    assert split.lower() in [
        "train",
        "val",
        "validation",
        "test",
    ], "Unknown split type!"

    if dataset_name.lower() == "pannuke":
        if split == "train":
            folds = dataset_config["train_folds"]
            dataset = PanNukeDataset(
                dataset_path=dataset_config["dataset_path"],
                folds=folds,
                transforms=transforms,
                stardist=dataset_config.get("stardist", False),
                regression=dataset_config.get("regression_loss", False),
                density=dataset_config.get("den_loss", False),
                is_train=True,
            )            
        if split == "val" or split == "validation":
            folds = dataset_config["val_folds"]
            dataset = PanNukeDataset(
                dataset_path=dataset_config["dataset_path"],
                folds=folds,
                transforms=transforms,
                stardist=dataset_config.get("stardist", False),
                regression=dataset_config.get("regression_loss", False),
                density=dataset_config.get("den_loss", False),
                is_train=False,
            )  
        if split == "test":
            folds = dataset_config["test_folds"]
            dataset = PanNukeDataset(
                dataset_path=dataset_config["dataset_path"],
                folds=folds,
                transforms=transforms,
                stardist=dataset_config.get("stardist", False),
                regression=dataset_config.get("regression_loss", False),
                density=dataset_config.get("den_loss", False),
                is_train=False,
            )

    else:
        raise NotImplementedError(f"Unknown dataset: {dataset_name}")
    return dataset
