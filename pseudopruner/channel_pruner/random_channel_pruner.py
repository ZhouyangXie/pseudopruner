import torch

from .channel_pruner import ChannelPruner


class RandomChannelPruner(ChannelPruner):
    def compute_mask(self, model):
        for name, module in model.named_modules():
            if not hasattr(module, 'to_prune'):
                continue

            if not module.to_prune:
                continue

            assert hasattr(module, 'sparsity')
            assert 0 <= float(module.sparsity) <= 1.0

            rand = torch.rand(module.prune_channel_mask.shape)
            to_mask = rand < module.sparsity
            module.prune_channel_mask[to_mask] = True
