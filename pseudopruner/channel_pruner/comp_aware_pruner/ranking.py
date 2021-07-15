import numpy as np
from scipy.linalg import block_diag
from scipy.linalg.lapack import dtrtri


def rank_channels(module):
    weight = module.weight.clone()
    if hasattr(module, 'prune_weight_mask'):
        weight[module.prune_weight_mask, ] = 0

    return _rank_channels_linear_conv(weight, module.mu, module.sigma)


def _low_tri_inv(L):
    L_inv, info = dtrtri(L, lower=1)
    if info == 0:
        return L_inv
    else:
        raise RuntimeError(f'lapack.dtrtri: error code {info}')


def _torch2numpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.float64)


def _rank_channels_linear_conv(weight, mu, sigma):
    weight = _torch2numpy(weight)
    if weight.ndim == 2:
        weight = weight.reshape(weight.shape + (1, 1))
    mu = _torch2numpy(mu)
    sigma = _torch2numpy(sigma)

    sigma_cc = sigma - np.outer(mu, mu)
    # set zero-var channels to a minimal positive variance
    for i, sigma_ii in enumerate(sigma_cc.diagonal()):
        if sigma_ii <= 0:
            sigma_cc[i, i] = 1e-20

    N, C, kw, kh = weight.shape
    dim = len(mu)
    assert dim == C * kw * kh
    ks = kw * kh

    weight = weight.reshape((N, dim))  # NxDim
    w_sigma_cc = weight.dot(sigma_cc)

    ranks = []
    # L_s_inv = None
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
                    L_t = np.linalg.cholesky(sigma_tt)
                    L_t_inv = _low_tri_inv(L_t)
                except (np.linalg.LinAlgError, RuntimeError):
                    # numerically unstable
                    continue

                score = w_sigma_ct.dot(L_t_inv.T)
                score = np.square(score).sum()

                if score > max_score:
                    max_score = score
                    max_ind = j
                    max_sigma_ss_inv = L_t_inv.T.dot(L_t_inv)

            sigma_ss_inv = max_sigma_ss_inv
            assert sigma_ss_inv.shape == (ks, ks)
            ranks.append(max_ind)

        else:
            non_rank = [
                r for r in range(C) if r not in ranks
            ]
            s_inds = np.concatenate([
                np.arange(ks*selected_i, ks*(selected_i + 1))
                for selected_i in ranks
            ])  # in selection order
            p_inds = np.concatenate([
                np.arange(ks*unselected, ks*(unselected + 1))
                for unselected in non_rank
            ])  # incremental

            sigma_sc = sigma_cc[s_inds, :]

            sigma_sp = sigma_sc[:, p_inds]
            SssinvSsp = sigma_ss_inv.dot(sigma_sp)
            SpsSssinvSsp = sigma_sp.T.dot(SssinvSsp)

            w_sigma_cp = w_sigma_cc[:, p_inds]
            w_sigma_cs = w_sigma_cc[:, s_inds]
            score_base_np = w_sigma_cp - w_sigma_cs.dot(SssinvSsp)

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
                    A_inv = np.linalg.cholesky(temp_tt)
                    A = _low_tri_inv(A_inv)
                except np.linalg.LinAlgError:
                    # numerically unstable
                    raise RuntimeWarning
                    continue

                score = score_base_np[:, ks*j_ind:(ks*j_ind+ks)]
                score = score.dot(A.T)
                score = np.square(score).sum()

                if score > max_score:
                    max_score = score
                    max_ind = j
                    max_j_ind = j_ind
                    max_A = A.copy()

            SssinvSsmax = SssinvSsp[:, (max_j_ind*ks):((max_j_ind+1)*ks)]
            ATA = max_A.T.dot(max_A)  # ks x ks
            RTA = - SssinvSsmax.dot(ATA)  # S x ks
            RTR = - RTA.dot(SssinvSsmax.T)  # S x ks
            sigma_ss_inv = block_diag(
                sigma_ss_inv + RTR, ATA
            )
            sigma_ss_inv[-ks:, 0:-ks] = RTA.T
            sigma_ss_inv[0:-ks, -ks:] = RTA

            ranks.append(max_ind)

    return ranks
