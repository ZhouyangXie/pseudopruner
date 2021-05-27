import logging

import torch
import numpy as np

from .channel_pruner import ChannelPruner

_logger = logging.getLogger(__name__)


class CorrChannelPruner(ChannelPruner):
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
                kw, kh = module.kernel_size
                in_channels = module.in_channels
                dim = in_channels * kw * kh

                sigma = module.sigma.detach().cpu().numpy()
                mu = module.mu.detach().cpu().numpy()
                assert sigma.shape == (dim, dim)
                assert mu.shape == (dim, )

                covar = sigma - np.outer(mu, mu)
                var = np.diag(covar).copy()

                var[var == 0] = 1
                corr = covar/np.sqrt(np.outer(var, var))

                # obtain only corr(x_i, x_j) i < j
                corr = np.triu(np.abs(corr))
                for i in range(len(corr)):
                    corr[i, i] = 0
                # higher max_corr means more replacible
                replacible = corr.max(axis=1)
                # 0-variance inputs are replacible
                replacible[replacible == 0] = 1

                # prune the channels with min-max correlation
                max_replacible = replacible.reshape((in_channels, kw, kh))
                minmax_replacible = max_replacible.min(axis=(1, 2))

            elif isinstance(module, torch.nn.Linear):
                raise NotImplementedError
            else:
                raise TypeError

            assert hasattr(module, 'mincorr')
            to_mask = minmax_replacible > module.mincorr
            module.prune_channel_mask[to_mask, ] = True

            num_pruned = to_mask.sum()
            _logger.debug(
                f'module {name}: mask {num_pruned}/{len(to_mask)} channels'
            )
