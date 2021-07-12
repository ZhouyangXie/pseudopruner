import numpy as np
from scipy.linalg.lapack import dpotri

import torch
from torch.nn.parameter import Parameter


def _inv_psd(A):
    L = np.linalg.cholesky(A)
    Ainv, info = dpotri(L, lower=True)
    if info > 0:
        raise RuntimeError(f'info from dpotri is {info}')
    Ainv += np.tril(Ainv, k=-1).T
    return Ainv


def update_weight(model):
    for name, module in model.named_modules():
        if hasattr(module, 'to_prune') and module.to_prune:
            assert hasattr(module, 'sample_counter') and \
                hasattr(module, 'mu') and \
                hasattr(module, 'sigma'), \
                "should have performed feautre statistics before compensation"
            if isinstance(module, torch.nn.Conv2d):
                _update_conv_weight(module)
            elif isinstance(module, torch.nn.Linear):
                _update_linear_weight(module)
            else:
                raise ValueError


def compute_conv_rec_loss(module):
    assert isinstance(module, torch.nn.Conv2d)
    kh, kw = module.kernel_size
    in_channels = module.in_channels
    out_channels = module.out_channels

    # obtain and check statistics and masks
    mu = module.mu.detach().cpu().numpy()
    C = len(mu)
    assert C == in_channels * kh * kw
    sigma = module.sigma.detach().cpu().numpy()
    assert sigma.shape == (C, C)
    # True if gonna be removed
    channel_mask = module.prune_channel_mask.detach().cpu().numpy()
    assert channel_mask.shape == (in_channels, )
    mask = np.repeat(channel_mask, kw * kh)
    assert mask.shape == (C, )
    # True if selected
    s_mask = ~mask
    S = int(s_mask.sum())
    W_old = module.weight.detach().cpu().numpy().reshape(
        (out_channels, C)
    )
    # skip unpruned layers
    if S == C:
        return 0
    mu_s = mu[s_mask]
    sigma_ss = sigma[s_mask][:, s_mask]
    sigma_cs = sigma[:, s_mask]

    A = sigma_ss - np.outer(mu_s, mu_s)
    B = np.dot(
            sigma_cs.T - np.outer(mu_s, mu),
            W_old.T
        )
    assert B.shape == (S, out_channels)
    try:
        A_inv = _inv_psd(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.inv(A)

    # compute reconstruction loss
    first_term = (W_old.dot(sigma - np.outer(mu, mu)) * W_old).sum()
    second_term = (A_inv.dot(B) * B).sum()
    return first_term - second_term


def _update_linear_weight(module):
    assert isinstance(module, torch.nn.Linear)
    in_features = module.in_channels
    out_features = module.out_channels

    # obtain and check statistics and masks
    mu = module.mu.detach().cpu().numpy()
    C = len(mu)
    assert C == in_features
    sigma = module.sigma.detach().cpu().numpy()
    assert sigma.shape == (C, C)
    # True if gonna be removed
    mask = module.prune_channel_mask.detach().cpu().numpy()
    assert mask.shape == (C, )
    # True if selected
    s_mask = ~mask
    S = int(s_mask.sum())
    W_old = module.weight.detach().cpu().numpy()
    assert W_old.shape == (out_features, C)
    # skip unpruned layers
    if S == C:
        return

    # compute statistics of selected channels
    mu_s = mu[s_mask]
    sigma_ss = sigma[s_mask][:, s_mask]
    sigma_cs = sigma[:, s_mask]
    assert mu_s.shape == (S, )
    assert sigma_ss.shape == (S, S)
    assert sigma_cs.shape == (C, S)

    # AW' = B
    A = sigma_ss - np.outer(mu_s, mu_s)
    assert A.sum() > 0  # not negative-definite
    B = np.dot(
            sigma_cs.T - np.outer(mu_s, mu),
            W_old.T
        )
    assert B.shape == (S, out_features)
    A_inv = _inv_psd(A)
    W_new = A_inv.dot(B)
    assert W_new.shape == (S, out_features)
    W_new = W_new.transpose()

    # rescale output variance!!!
    Sigma_cc = sigma - np.outer(mu, mu)
    Sigma_ss = A
    var_old = (W_old.dot(Sigma_cc) * W_old).sum(axis=1)  # N
    var_new = (W_new.dot(Sigma_ss) * W_new).sum(axis=1)  # N
    scaling = np.sqrt(var_old/var_new)
    W_new = W_new * scaling[:, None]
    assert W_new.shape == (out_features, S)
    # end

    if module.bias is not None:
        bias_old = module.bias.detach().cpu().numpy()
    else:
        bias_old = np.zeros(out_features, dtype=np.float32)
        module.register_parameter('bias', Parameter(
            torch.from_numpy(bias_old)
        ))
        module.bias.requires_grad = False
        module.bias.to(module.weight.device)

    bias_new = bias_old - \
        np.dot(mu_s, W_new.T) + \
        np.dot(mu, W_old.T)

    # assign new weights and bias
    module.weight[:, ~mask] = \
        torch.from_numpy(
            W_new.astype(np.float32)
        ).to(module.weight.device)

    module.bias[:] = torch.from_numpy(
        bias_new.astype(np.float32)).to(module.bias.device)

    # assign zero to masked_weight
    module.weight[:, mask] = 0


def _update_conv_weight(module):
    '''
        Experimental updating plan for convs followed by a BN
        The variance of the pruned output is rescaled to the old variance
        While recontruction loss is a little bit undermined
    '''
    assert isinstance(module, torch.nn.Conv2d)
    kh, kw = module.kernel_size
    in_channels = module.in_channels
    out_channels = module.out_channels

    # obtain and check statistics and masks
    mu = module.mu.detach().cpu().numpy()
    C = len(mu)
    assert C == in_channels * kh * kw
    sigma = module.sigma.detach().cpu().numpy()
    assert sigma.shape == (C, C)
    # True if gonna be removed
    channel_mask = module.prune_channel_mask.detach().cpu().numpy()
    assert channel_mask.shape == (in_channels, )
    mask = np.repeat(channel_mask, kw * kh)
    assert mask.shape == (C, )
    # True if selected
    s_mask = ~mask
    S = int(s_mask.sum())
    W_old = module.weight.detach().cpu().numpy().reshape(
        (out_channels, C)
    )
    # skip unpruned layers
    if S == C:
        return

    # compute statistics of selected channels
    mu_s = mu[s_mask]
    sigma_ss = sigma[s_mask][:, s_mask]
    sigma_cs = sigma[:, s_mask]
    assert mu_s.shape == (S, )
    assert sigma_ss.shape == (S, S)
    assert sigma_cs.shape == (C, S)

    # AW' = B
    A = sigma_ss - np.outer(mu_s, mu_s)
    assert A.sum() > 0  # not negative-definite
    B = np.dot(
            sigma_cs.T - np.outer(mu_s, mu),
            W_old.T
        )
    assert B.shape == (S, out_channels)
    A_inv = _inv_psd(A)
    W_new = A_inv.dot(B)
    assert W_new.shape == (S, out_channels)
    W_new = W_new.transpose()

    # rescale output variance!!!
    Sigma_cc = sigma - np.outer(mu, mu)
    Sigma_ss = A
    var_old = (W_old.dot(Sigma_cc) * W_old).sum(axis=1)  # N
    var_new = (W_new.dot(Sigma_ss) * W_new).sum(axis=1)  # N
    scaling = np.sqrt(var_old/var_new)
    W_new = W_new * scaling[:, None]
    assert W_new.shape == (out_channels, S)
    # end

    if module.bias is not None:
        bias_old = module.bias.detach().cpu().numpy()
    else:
        bias_old = np.zeros(out_channels, dtype=np.float32)
        module.register_parameter('bias', Parameter(
            torch.from_numpy(bias_old)
        ))
        module.bias.requires_grad = False
        module.bias.to(module.weight.device)

    bias_new = bias_old - \
        np.dot(mu_s, W_new.T) + \
        np.dot(mu, W_old.T)

    W_new_expanded = W_new.reshape(
        (out_channels, int((~channel_mask).sum()), kw, kh))

    # assign new weights and bias
    module.weight[:, ~channel_mask, :, :] = \
        torch.from_numpy(
            W_new_expanded.astype(np.float32)
        ).to(module.weight.device)

    module.bias[:] = torch.from_numpy(
        bias_new.astype(np.float32)).to(module.bias.device)

    # assign zero to masked_weight
    module.weight[:, channel_mask, :, :] = 0
