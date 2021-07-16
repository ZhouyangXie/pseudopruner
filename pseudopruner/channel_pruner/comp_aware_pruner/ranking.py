import torch
from torch.nn import Linear, Conv2d
import numpy as np
from scipy.linalg.lapack import dtrtri

import logging


def rank_channels(module, device='cuda'):
    weight = module.weight.clone()
    if hasattr(module, 'prune_weight_mask'):
        weight[module.prune_weight_mask, ] = 0

    if isinstance(module, Linear):
        return _rank_channels_linear(weight, module.mu, module.sigma, device)
    elif isinstance(module, Conv2d):
        return _rank_channels_conv(weight, module.mu, module.sigma, device)
    else:
        raise NotImplementedError


def _low_tri_inv(L):
    L_inv, info = dtrtri(L, lower=1)
    if info == 0:
        return L_inv
    else:
        raise RuntimeError(f'lapack.dtrtri: error code {info}')


def _torch2numpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.float64)


def _numpy2torch(ndarray):
    return torch.from_numpy(ndarray).to(
        device='cuda:0', dtype=torch.float64)


def _rank_channels_conv(weight, mu, sigma, device='cuda'):
    N, C, kw, kh = weight.shape
    dim = len(mu)
    assert dim == C * kw * kh
    ks = kw * kh

    if ks == 1:
        weight = weight.reshape((N, C))
        return _rank_channels_linear(weight, mu, sigma)

    weight = weight.to(device=device, dtype=torch.float64)
    mu = mu.to(device=device, dtype=torch.float64)
    sigma = sigma.to(device=device, dtype=torch.float64)

    sigma_cc = sigma - torch.outer(mu, mu)
    # set zero-var channels to a minimal positive variance
    for i in range(dim):
        if sigma_cc[i, i] <= 0:
            sigma_cc[i, i] = 1e-20

    weight = weight.reshape((N, dim))  # N x Dim
    w_sigma_cc = weight.matmul(sigma_cc)

    ranks = []
    sigma_ss_inv = None

    for i in range(C):
        # if the first round
        if sigma_ss_inv is None:
            max_score = -1
            max_ind = None
            max_sigma_ss_inv = None
            for j in range(C):
                w_sigma_ct = w_sigma_cc[:, (j*ks):((j+1)*ks)]
                sigma_tt = sigma_cc[(j*ks):((j+1)*ks), (j*ks):((j+1)*ks)]

                try:
                    sigma_tt = _torch2numpy(sigma_tt)
                    L_t = np.linalg.cholesky(sigma_tt)
                    L_t_inv = _low_tri_inv(L_t)
                    L_t_inv = _numpy2torch(L_t_inv)
                except (np.linalg.LinAlgError, RuntimeError):
                    # numerically unstable
                    continue

                score = w_sigma_ct.matmul(L_t_inv.T)
                score = float(torch.square(score).sum())

                if score > max_score:
                    max_score = score
                    max_ind = j
                    max_sigma_ss_inv = L_t_inv.T.matmul(L_t_inv)

            sigma_ss_inv = max_sigma_ss_inv
            assert sigma_ss_inv.shape == (ks, ks)
            ranks.append(max_ind)

        else:
            non_rank = [
                r for r in range(C) if r not in ranks
            ]
            s_inds = torch.cat([
                torch.arange(
                    ks*selected_i, ks*(selected_i + 1), device=device)
                for selected_i in ranks
            ])  # in selection order
            p_inds = torch.cat([
                torch.arange(
                    ks*unselected, ks*(unselected + 1), device=device)
                for unselected in non_rank
            ])  # incremental

            # numerical error check
            # sigma_ss = sigma_cc[s_inds][:, s_inds]
            # iden = sigma_ss_inv.matmul(sigma_ss)
            # eye = torch.eye(
            #   iden.shape[0], device=device, dtype=torch.float64)
            # err = torch.abs(iden - eye).max()
            # logging.info(
            #     f'Numerical abs error of iden:{float(err)}')

            sigma_sc = sigma_cc[s_inds, :]

            sigma_sp = sigma_sc[:, p_inds]
            SssinvSsp = sigma_ss_inv.matmul(sigma_sp)
            SpsSssinvSsp = sigma_sp.T.matmul(SssinvSsp)

            w_sigma_cp = w_sigma_cc[:, p_inds]
            w_sigma_cs = w_sigma_cc[:, s_inds]
            score_base_np = w_sigma_cp - w_sigma_cs.matmul(SssinvSsp)

            max_score = -1
            max_ind = None
            max_j_ind = None
            max_A = None

            for j_ind, j in enumerate(non_rank):
                sigma_tt = sigma_cc[(j*ks):((j+1)*ks), (j*ks):((j+1)*ks)]
                temp_tt = sigma_tt - SpsSssinvSsp[
                    (j_ind*ks):((j_ind+1)*ks), (j_ind*ks):((j_ind+1)*ks)
                ]
                try:
                    temp_tt = _torch2numpy(temp_tt)
                    A_inv = np.linalg.cholesky(temp_tt)
                    A = _low_tri_inv(A_inv)
                    A = _numpy2torch(A)
                except np.linalg.LinAlgError:
                    # numerically unstable
                    continue

                score = score_base_np[:, ks*j_ind:(ks*j_ind+ks)]
                score = score.matmul(A.T)
                score = float(torch.square(score).sum())

                if score > max_score:
                    max_score = score
                    max_ind = j
                    max_j_ind = j_ind
                    max_A = A.clone()

            if max_ind is None:
                ranks = ranks + non_rank
                logging.warning(
                    f'Numerical error. Channels after {i}/{C} are un-ranked')
                break

            SssinvSsmax = SssinvSsp[:, (max_j_ind*ks):((max_j_ind+1)*ks)]
            ATA = max_A.T.matmul(max_A)  # ks x ks
            RTA = - SssinvSsmax.matmul(ATA)  # S x ks
            RTR = - RTA.matmul(SssinvSsmax.T)  # S x ks
            sigma_ss_inv = torch.block_diag(
                sigma_ss_inv + RTR, ATA
            )
            sigma_ss_inv[-ks:, 0:-ks] = RTA.T
            sigma_ss_inv[0:-ks, -ks:] = RTA

            ranks.append(max_ind)

        # logging.info(f'{i}/{C}: {ranks[-1]}')

    return ranks


def _rank_channels_linear(weight, mu, sigma, device='cuda'):
    weight = weight.to(device=device, dtype=torch.float64)
    N, C = weight.shape

    mu = mu.to(device=device, dtype=torch.float64)
    sigma = sigma.to(device=device, dtype=torch.float64)

    sigma_cc = sigma - torch.outer(mu, mu)
    # set zero-var channels to a minimal positive variance
    for i in range(C):
        if sigma_cc[i, i] <= 0:
            sigma_cc[i, i] = 1e-20

    w_sigma_cc = weight.matmul(sigma_cc)
    ranks = []
    sigma_ss_inv = None
    sigma_c = sigma_cc.diag()

    for i in range(C):
        # if the first round
        if sigma_ss_inv is None:
            scores = torch.square(w_sigma_cc)
            scores = scores.sum(axis=0)
            scores = scores/sigma_c
            max_ind = int(torch.argmax(scores))
            ranks.append(max_ind)
            sigma_ss_inv = torch.ones(
                (1, 1), device=device, dtype=torch.float64)
            sigma_ss_inv[:] = 1/sigma_c[max_ind]

        else:
            non_ranks = [
                r for r in range(C) if r not in ranks
            ]
            s_inds = torch.tensor(ranks, device=device, dtype=torch.int64)
            p_inds = torch.tensor(non_ranks, device=device, dtype=torch.int64)

            # numerical error check
            # sigma_ss = sigma_cc[s_inds][:, s_inds]
            # iden = sigma_ss_inv.matmul(sigma_ss)
            # eye = torch.eye(
            #   iden.shape[0], device=device, dtype=torch.float64)
            # err = torch.abs(iden - eye).max()
            # logging.info(
            #     f'Numerical abs error of iden:{float(err)}')

            sigma_ts = sigma_cc[p_inds][:, s_inds]
            StsSssinv = sigma_ts.matmul(sigma_ss_inv)

            # compute $a^2$ among T
            sigma_t = sigma_c[p_inds]
            temp = (StsSssinv * sigma_ts).sum(axis=1)
            A_square = 1/(sigma_t - temp)

            # compute score
            w_sigma_ct = w_sigma_cc[:, p_inds]
            w_sigma_cs = w_sigma_cc[:, s_inds]
            scores = w_sigma_ct - w_sigma_cs.matmul(StsSssinv.T)
            scores = torch.square(scores).sum(axis=0)
            scores = scores * A_square

            max_ind = int(torch.argmax(scores))

            # compute new \sigma_{S, S}^{-1}
            a_square = A_square[max_ind]
            if float(a_square) <= 0 or torch.isnan(a_square).any():
                ranks = ranks + non_ranks
                logging.warning(
                    f'Numerical error. Channels after {i}/{C} are un-ranked')
                break

            sigma_ss_inv = torch.block_diag(
                sigma_ss_inv, torch.zeros(
                    (1, 1), dtype=torch.float64, device=device)
            )
            row = torch.cat(
                [
                    StsSssinv[max_ind, :],
                    -torch.ones((1,), dtype=torch.float64, device=device)
                ]
            )
            mat = torch.outer(row, row)
            mat = mat * a_square
            sigma_ss_inv += mat
            # r_div_a = - StsSssinv[max_ind, :]
            # sigma_ss_inv = torch.block_diag(
            #     sigma_ss_inv + a_square * torch.outer(r_div_a, r_div_a),
            #     a_square
            # )
            # a_r = a_square * r_div_a
            # sigma_ss_inv[-1, :-1] = a_r.flatten()
            # sigma_ss_inv[:-1, -1] = a_r.flatten()

            ranks.append(non_ranks[max_ind])
            # logging.info(
            #     f'a^2:{float(a_square)}')

        # logging.info(f'{i}/{C}: {ranks[-1]}')

    return ranks
