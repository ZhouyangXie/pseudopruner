import torch

from .weight_pruner import WeightPruner


class FPGMPruner(WeightPruner):
    def compute_mask(self, model):
        for name, module in model.named_modules():
            if not hasattr(module, 'to_prune'):
                continue
            if not module.to_prune:
                continue
            if not isinstance(module, torch.nn.Conv2d):
                continue

            assert hasattr(module, 'prune_weight_mask')
            assert hasattr(module, 'sparsity')
            sparsity = float(module.sparsity)
            assert 0 <= sparsity < 1
            num_pruned = int(sparsity * module.out_channels)
            if num_pruned <= 0:
                continue

            ranks = FPGMPruner._get_min_gm_kernel_idx(module.weight.data)
            ranks = ranks.to(module.prune_weight_mask.device)
            module.prune_weight_mask[ranks[-num_pruned:], ] = True

    @staticmethod
    def _get_min_gm_kernel_idx(weight):
        channel_dist = FPGMPruner.get_channel_sum(weight)
        dist_list = [(channel_dist[i], i)
                     for i in range(channel_dist.size(0))]
        min_gm_kernels = sorted(dist_list, key=lambda x: x[0], reverse=True)
        return [x[1] for x in min_gm_kernels]

    @staticmethod
    def get_channel_sum(weight):
        assert len(weight.size()) in [3, 4]
        dist_list = []
        for out_i in range(weight.size(0)):
            dist_sum = FPGMPruner._get_distance_sum(weight, out_i)
            dist_list.append(dist_sum)
        return torch.Tensor(dist_list).to(weight.device)

    @staticmethod
    def _get_distance_sum(weight, out_idx):
        assert len(weight.size()) in [3, 4], 'unsupported weight shape'

        w = weight.view(weight.size(0), -1)
        anchor_w = w[out_idx].unsqueeze(0).expand(w.size(0), w.size(1))
        x = w - anchor_w
        x = (x * x).sum(-1)
        x = torch.sqrt(x)
        return x.sum()
