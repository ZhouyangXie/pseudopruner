import torch

from .channel_pruner import ChannelPruner


class NormChannelPruner(ChannelPruner):
    def compute_mask(self, model):
        for name, module in model.named_modules():
            if not hasattr(module, 'to_prune') or\
               not module.to_prune or\
               not isinstance(module, torch.nn.Conv2d):
                continue

            assert hasattr(module, 'prune_channel_mask')
            assert hasattr(module, 'sparsity')

            sparsity = float(module.sparsity)
            assert 0.0 <= sparsity < 1.0

            kw, kh = module.kernel_size
            in_channels = module.in_channels
            dim = in_channels * kw * kh

            num_masked = int(in_channels * sparsity)
            if num_masked == 0:
                continue

            weight = module.weight.clone()
            scores = weight.square().sum(axis=(0, 2, 3))
            min_score_inds = torch.argsort(scores)[0:num_masked]
            module.prune_channel_mask[min_score_inds, ] = True
