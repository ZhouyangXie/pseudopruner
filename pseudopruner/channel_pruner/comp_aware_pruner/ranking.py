import torch


def rank_channels(module, device):
    weight = module.weight.clone()
    if hasattr(module, 'prune_weight_mask'):
        weight[module.prune_weight_mask, ] = 0

    if isinstance(module, torch.nn.Conv2d):
        return _rank_channels_conv(weight, module.mu, module.sigma, device)
    elif isinstance(module, torch.nn.Linear):
        return _rank_channels_linear(weight, module.mu, module.sigma, device)
    else:
        raise NotImplementedError


def _rank_channels_conv(weight, mu, sigma, device):
    sigma_cc = sigma - torch.outer(mu, mu)
    sigma_cc = sigma_cc.to(
        dtype=torch.float32, device=device)

    # set zero-var channels to a minimal positive variance
    for i in range(len(sigma_cc)):
        if sigma_cc[i, i] == 0:
            sigma_cc[i, i] += 1e-20

    N, out_channels, kw, kh = weight.shape
    dim = len(mu)
    assert dim == out_channels * kw * kh

    weight = weight.to(device=device, dtype=torch.float32)
    weight = weight.reshape((N, -1))  # NxDim
    D = weight.matmul(sigma_cc.to(device))

    selected = torch.zeros(dim, dtype=bool, device=device)
    all_inds = torch.arange(dim)
    S = []
    L_s_inv = None

    for i in range(dim):
        torch.cuda.empty_cache()
        # if the first round
        if L_s_inv is None:
            sigma_all_t = sigma_cc.diag()
            D_all_t = torch.square(D).sum(axis=0)
            scores = D_all_t/sigma_all_t
            max_j = int(torch.argmax(scores))

            selected[max_j] = True
            S.append(max_j)
            L_s_inv = torch.ones((1, 1), device=device, dtype=torch.float32)
            L_s_inv[:] = 1/sigma_all_t[max_j].sqrt()

        else:
            T = ~selected
            S_cuda = torch.tensor(S, device=device, dtype=torch.long)

            sigma_tt_diag = sigma_cc.diag()[T]
            sigma_st = sigma_cc[S_cuda][:, T]

            D_s = D[:, S_cuda]  # N X S
            D_t = D[:, T]  # N X T

            b = L_s_inv.matmul(sigma_st)  # S X T
            bbt = (b * b).sum(axis=0)

            r = torch.sqrt(sigma_tt_diag - bbt)  # T
            rows = -(L_s_inv.T.matmul(b))/r  # S X T

            scores = D_s.matmul(rows) + D_t/r  # N X T
            scores = torch.square(scores).sum(axis=0)  # T

            max_j = torch.argmax(scores)
            assert not torch.isnan(scores).any()
            assert not torch.isinf(scores).any()

            max_r = r[max_j].reshape((1,))
            max_row = rows[:, max_j].flatten()
            L_s_inv = torch.block_diag(L_s_inv, 1/max_r)
            L_s_inv[-1, 0:-1] = max_row

            new_s = int(all_inds[T][max_j])
            selected[new_s] = True
            S.append(new_s)

    return S


def _rank_channels_linear(weight, mu, sigma, device):
    sigma_cc = sigma - torch.outer(mu, mu)
    sigma_cc = sigma_cc.to(
        dtype=torch.float32, device=device)

    # set zero-var channels to a minimal positive variance
    for i in range(len(sigma_cc)):
        if sigma_cc[i, i] == 0:
            sigma_cc[i, i] += 1e-20

    N, _dim = weight.shape
    dim = len(mu)
    assert dim == _dim
    assert sigma_cc.shape == (dim, dim)

    # N X dim
    weight = weight.to(device=device, dtype=torch.float32)
    # N X dim
    D = weight.matmul(sigma_cc.to(device))

    selected = torch.zeros(dim, dtype=bool, device=device)
    all_inds = torch.arange(dim)
    S = []
    L_s_inv = None

    for i in range(dim):
        torch.cuda.empty_cache()
        # if the first round
        if L_s_inv is None:
            sigma_all_t = sigma_cc.diag()
            D_all_t = torch.square(D).sum(axis=0)
            scores = D_all_t/sigma_all_t
            max_j = int(torch.argmax(scores))

            selected[max_j] = True
            S.append(max_j)
            L_s_inv = torch.ones((1, 1), device=device, dtype=torch.float32)
            L_s_inv[:] = 1/sigma_all_t[max_j].sqrt()

        else:
            T = ~selected
            S_cuda = torch.tensor(S, device=device, dtype=torch.long)

            sigma_tt_diag = sigma_cc.diag()[T]
            sigma_st = sigma_cc[S_cuda][:, T]

            D_s = D[:, S_cuda]  # N X S
            D_t = D[:, T]  # N X T

            b = L_s_inv.matmul(sigma_st)  # S X T
            bbt = (b * b).sum(axis=0)

            r = torch.sqrt(sigma_tt_diag - bbt)  # T
            rows = -(L_s_inv.T.matmul(b))/r  # S X T

            scores = D_s.matmul(rows) + D_t/r  # N X T
            scores = torch.square(scores).sum(axis=0)  # T

            max_j = torch.argmax(scores)
            assert not torch.isnan(scores).any()
            assert not torch.isinf(scores).any()

            max_r = r[max_j].reshape((1,))
            max_row = rows[:, max_j].flatten()
            L_s_inv = torch.block_diag(L_s_inv, 1/max_r)
            L_s_inv[-1, 0:-1] = max_row

            new_s = int(all_inds[T][max_j])
            selected[new_s] = True
            S.append(new_s)

    return S
