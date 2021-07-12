import logging
from time import perf_counter
import torch


_script_name = __file__.strip('.py')
logging.basicConfig(
    filename=f'{_script_name}.log',
    filemode='w',
    format='%(asctime)s | %(message)s',
    datefmt='%d/%m %H:%M:%S',
    level=logging.DEBUG
)
_device = 'cuda:0'


class CuTimer:
    def __init__(self):
        self.reset(True)

    def checkpoint(self, msg=''):
        if self.enabled:
            torch.cuda.synchronize(_device)
            if self.start_time:
                self.events.append(
                    (msg, perf_counter() - self.start_time)
                )

            self.start_time = perf_counter()

    def reset(self, enabled):
        self.enabled = enabled
        self.start_time = None
        self.events = []

    def log_event(self):
        for msg, elapsed in self.events:
            logging.debug(f'event {msg}: {elapsed:.6f}s')


def main():
    ds = torch.load('resnet50_stat.pt', 'cpu')
    layer = 'layer4.1.conv2.'  # 512 X 512 X 3 X 3
    # layer = 'layer4.1.conv1.'  # 512 X 2048 X 1 X 1
    weight = ds[layer + 'weight']
    sigma = ds[layer + 'sigma']
    mu = ds[layer + 'mu']
    sigma_cc = sigma - torch.outer(mu, mu)
    sigma_cc = sigma_cc.to(
        dtype=torch.float32, device=_device)

    N, out_channels, kw, kh = weight.shape
    dim = len(mu)
    assert dim == out_channels * kw * kh

    logging.debug(f'rank channels {layer}: {N} X {out_channels} X {kw} X {kh}')
    start = perf_counter()

    weight = weight.to(device=_device, dtype=torch.float32)
    weight = weight.reshape((N, -1))  # NxDim
    D = weight.matmul(sigma_cc.to(_device))

    # logging.info(f'dim: {out_channels} X {dim}')

    selected = torch.zeros(dim, dtype=bool, device=_device)
    all_inds = torch.arange(dim)
    S = []
    # timer = CuTimer()
    # timer.reset(False)

    L_s_inv = None
    for i in range(dim):
        # if i % 50 == 0:
        #     logging.info(f'iter i={i}')
        torch.cuda.empty_cache()
        # if the first round
        if L_s_inv is None:
            sigma_all_t = sigma_cc.diag()
            D_all_t = torch.square(D).sum(axis=0)
            scores = D_all_t/sigma_all_t
            max_j = int(torch.argmax(scores))

            selected[max_j] = True
            S.append(max_j)
            L_s_inv = torch.ones((1, 1), device=_device, dtype=torch.float32)
            L_s_inv[:] = 1/sigma_all_t[max_j].sqrt()

        # that S is not empty
        else:
            T = ~selected

            # verification
            # sigma_ss = sigma_cc[S][:, S]
            # Iden = L_s_inv.T.matmul(L_s_inv).matmul(sigma_ss)
            # assert torch.allclose(
            #     Iden, torch.eye(len(S), device=_device, dtype=torch.float32),
            #     atol=1e-4)
            # verification
            # if i == 1000:
            #     timer.reset(True)
            #     # timer.checkpoint()

            S_cuda = torch.tensor(S, device=_device, dtype=torch.long)
            # timer.checkpoint('prepare S_cuda')

            sigma_tt_diag = sigma_cc.diag()[T]
            sigma_st = sigma_cc[S_cuda][:, T]
            # timer.checkpoint('prepare sigmas')

            D_s = D[:, S_cuda]  # N X S
            D_t = D[:, T]  # N X T
            # timer.checkpoint('prepare Dt')

            b = L_s_inv.matmul(sigma_st)  # S X T
            bbt = (b * b).sum(axis=0)
            # timer.checkpoint('compute bbt')

            r = torch.sqrt(sigma_tt_diag - bbt)  # T
            # assert (r > 0).all()
            # timer.checkpoint('compute r')

            rows = -(L_s_inv.T.matmul(b))/r  # S X T
            # timer.checkpoint('compute rows')

            scores = D_s.matmul(rows) + D_t/r  # N X T
            scores = torch.square(scores).sum(axis=0)  # T
            # timer.checkpoint('compute scores')

            max_j = torch.argmax(scores)
            # timer.checkpoint('find best')

            max_r = r[max_j].reshape((1,))
            max_row = rows[:, max_j].flatten()
            L_s_inv = torch.block_diag(L_s_inv, 1/max_r)
            L_s_inv[-1, 0:-1] = max_row
            # timer.checkpoint('update L_s_inv')

            # max_j is the indice in T
            # we calculate max_j in dim
            new_s = int(all_inds[T][max_j])
            # timer.checkpoint('compute new_s')

            selected[new_s] = True
            S.append(new_s)
            # timer.checkpoint('update S')

            # for verification
            # sigma_ss = sigma_cc[S][:, S]
            # true_L_s = torch.linalg.cholesky(sigma_ss)
            # true_L_s_inv = torch.linalg.inv(true_L_s)
            # true_v = true_L_s[-1, :-1]
            # true_r = true_L_s[-1, -1]
            # assert torch.allclose(b[:, max_j], true_v.T, atol=1e-4)
            # assert torch.allclose(true_r, max_r, atol=1e-4)
            # assert torch.allclose(true_L_s_inv, L_s_inv, atol=1e-4)
            # for verification

            # if timer.enabled:
            #     break

    # timer.log_event()
    elapse = perf_counter() - start
    logging.debug(f'rank channels {layer}: elapse {elapse}')
    logging.debug(S)


if __name__ == '__main__':
    main()
