from .channel_pruner import ChannelPruner

from .random_channel_pruner import RandomChannelPruner
from .const_channel_pruner import ConstChannelPruner
from .corr_channel_pruner import CorrChannelPruner
from .norm_pruner import NormChannelPruner

__all__ = [
    ChannelPruner.__name__,
    RandomChannelPruner.__name__,
    ConstChannelPruner.__name__,
    CorrChannelPruner.__name__,
    NormChannelPruner.__name__
]
