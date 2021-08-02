"""
    A structured pruning algorithm that prunes the filters with the smallest
    importance approximations based on the first order taylor expansion on the
    weight. "Importance Estimation for Neural Network Pruning", CVPR 2019.
    http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf
"""
import torch

from .weight_pruner import WeightPruner


class ImportEstPruner(WeightPruner):
    def compute_mask(self, model):
        for name, module in model.named_modules():
            if not hasattr(module, 'to_prune'):
                continue

            if not module.to_prune:
                continue

            assert hasattr(module, 'prune_weight_mask')
            assert hasattr(module, 'sparsity')
            assert hasattr(module, 'mu')
            sparsity = float(module.sparsity)
            assert 0 <= sparsity < 1

            if isinstance(module, torch.nn.Conv2d):
                num_pruned = int(sparsity * module.out_channels)
                weight = module.weight.data.reshape((module.out_channels, -1))
                mu = module.mu.reshape((-1))
            elif isinstance(module, torch.nn.Linear):
                num_pruned = int(sparsity * module.out_features)
                weight = module.weight.data
                mu = module.mu
            else:
                continue

            if num_pruned <= 0:
                continue

            scores = torch.square(weight * mu).sum(axis=1)
            min_score_inds = torch.argsort(scores)[0:num_pruned]
            min_score_inds = min_score_inds.to(module.prune_weight_mask.device)
            module.prune_weight_mask[min_score_inds, ] = True
