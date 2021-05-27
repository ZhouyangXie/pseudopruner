import torch
import numpy as np

from .channel_pruner import ChannelPruner


class ConstChannelPruner(ChannelPruner):
    def compute_mask(self, model):
        for name, module in model.named_modules():
            if not hasattr(module, 'to_prune'):
                continue

            if not module.to_prune:
                continue

            assert hasattr(module, 'prune_channel_mask')
            assert hasattr(module, 'sample_counter')
            assert hasattr(module, 'mu')
            assert hasattr(module, 'sigma')

            if isinstance(module, torch.nn.Conv2d):
                in_channels = module.in_channels
                kw, kh = module.kernel_size
                sigma = module.sigma.detach().cpu().numpy()
                mu = module.mu.detach().cpu().numpy()
                dim = in_channels * kw * kh
                assert sigma.shape == (dim, dim)
                assert mu.shape == (dim, )

                cov = sigma - np.outer(mu, mu)
                # mask the channels where all inputs are constant
                to_mask = np.diag(cov) == 0.0
                to_mask = to_mask.reshape((in_channels, kw, kh))
                to_mask = to_mask.all(axis=(1, 2))

            elif isinstance(module, torch.nn.Linear):
                raise NotImplementedError
            else:
                raise TypeError

            module.prune_channel_mask[to_mask, ] = True
