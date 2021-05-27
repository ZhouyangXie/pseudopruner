import logging

import torch

from .channel_pruner import ChannelPruner

_logger = logging.getLogger(__name__)


class RandomChannelPruner(ChannelPruner):
    def compute_mask(self, model):
        for name, module in model.named_modules():
            if not hasattr(module, 'to_prune'):
                continue

            if not module.to_prune:
                continue

            assert hasattr(module, 'sparsity')
            assert 0 <= float(module.sparsity) <= 1.0

            mask = module.prune_channel_mask
            rand = torch.rand(mask.shape)
            mask[:] = rand < module.sparsity

            num_masked = mask.sum()
            _logger.debug(
                f'module {name}: mask {int(num_masked)}/{len(mask)} channels'
            )
