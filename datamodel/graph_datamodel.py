from dataclasses import dataclass

import torch


@dataclass
class GraphDataWSI:

    x: torch.Tensor
    positions: torch.Tensor
    metadata: dict
