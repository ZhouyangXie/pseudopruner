import torch
import numpy as np

from .weight_pruner import WeightPruner


class NormPruner(WeightPruner):
    def compute_mask(self, model):
        for name, module in model.named_modules():
            if not hasattr(module, 'to_prune'):
                continue

            if not module.to_prune:
                continue

            assert hasattr(module, 'prune_weight_mask')
            assert hasattr(module, 'sparsity')
            sparsity = float(module.sparsity)
            assert 0 <= sparsity < 1

            if not hasattr(module, 'norm'):
                setattr(module, 'norm', 2)
            norm = int(module.norm)
            assert norm > 0

            if isinstance(module, torch.nn.Conv2d):
                num_pruned = int(sparsity * module.out_channels)
                weight = module.weight.cpu().detach().numpy()
                score = (np.abs(weight)**norm).sum(axis=(1, 2, 3))
            elif isinstance(module, torch.nn.Linear):
                raise NotImplementedError
            else:
                raise TypeError

            min_score_inds = np.argsort(score)[0:num_pruned]
            min_score_inds = torch.from_numpy(min_score_inds)
            module.prune_weight_mask[min_score_inds, ] = True
