import torch

from ..channel_pruner import ChannelPruner


class CompensationAwarePruner(ChannelPruner):
    def compute_mask(self, model):
        for name, module in model.named_modules():
            if not hasattr(module, 'to_prune') or\
               not module.to_prune or\
               not isinstance(module, torch.nn.Conv2d) or\
               not hasattr(module, 'ranks'):
                continue

            assert hasattr(module, 'prune_channel_mask')
            assert hasattr(module, 'sparsity')

            sparsity = float(module.sparsity)
            assert 0.0 <= sparsity < 1.0

            kw, kh = module.kernel_size
            in_channels = module.in_channels
            dim = in_channels * kw * kh
            ranks = module.ranks
            assert len(ranks) == dim

            num_masked = int(dim * sparsity)
            if num_masked == 0:
                continue

            # choose the first a few channels with important input be preserved
            to_keep = torch.zeros((in_channels, ), dtype=torch.bool)
            num_kept = int(in_channels * (1 - sparsity))
            for i in ranks:
                to_keep[i//(kw*kh)] = True
                if to_keep.sum() >= num_kept:
                    break

            module.prune_channel_mask[~to_keep, ] = True
