
from typing import Callable, Tuple, List

import torch
from torch.utils.data import Dataset
from datamodel.wsi_datamodel import WSI


class PatchedWSIInference(Dataset):

    def __init__(
        self,
        wsi_object: WSI,
        transform: Callable,
    ) -> None:
        # set all configurations
        assert isinstance(wsi_object, WSI), "Must be a WSI-object"
        assert (
            wsi_object.patched_slide_path is not None
        ), "Please provide a WSI that already has been patched into slices"

        self.transform = transform
        self.wsi_object = wsi_object

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, list[list[str, str]], list[str], int, str]:

        patch_name = self.wsi_object.patches_list[idx]

        patch, metadata = self.wsi_object.process_patch_image(
            patch_name=patch_name, transform=self.transform
        )

        return patch, metadata

    def __len__(self) -> int:

        return int(self.wsi_object.get_number_patches())

    @staticmethod
    def collate_batch(batch: List[Tuple]) -> Tuple[torch.Tensor, list[dict]]:

        patches, metadata = zip(*batch)
        patches = torch.stack(patches)
        metadata = list(metadata)
        return patches, metadata
