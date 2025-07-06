import logging
from typing import Callable
from torch.utils.data import Dataset

logger = logging.getLogger()
logger.addHandler(logging.NullHandler())



class nucleiDataset(Dataset):
    def set_transforms(self, transforms: Callable) -> None:
        self.transforms = transforms

