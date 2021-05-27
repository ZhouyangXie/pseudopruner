from abc import ABC, abstractmethod

import torch
import torch.nn


class ChannelPruner(ABC):
    _allow_module_types = [
        torch.nn.Conv2d,
        torch.nn.Linear
    ]

    def __init__(self):
        super().__init__()

    @abstractmethod
    def compute_mask(self, model):
        """
        Implemented by subclasses to compute channel masks. At this
            moment, to-be-pruned modules have 'prune_channel_mask'
            (all False) and some other necessary arguments registered.
            But the masks are not effective now.

        Args:
            model (torch.nn.Module): the model with modules marked as to-prune.
        """
        pass
