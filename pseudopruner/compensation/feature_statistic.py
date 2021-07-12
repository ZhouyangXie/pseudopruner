import torch
from torch.nn.functional import unfold


def register_statistics_hook(module, sampling_stride=1):
    '''
        record input's statistics of the module
    '''
    if isinstance(module, torch.nn.Conv2d):
        num_feature = module.in_channels * \
            module.kernel_size[0] * module.kernel_size[1]
        handle = module.register_forward_hook(_conv2d_statistics_hook)
        setattr(module, 'sampling_stride', sampling_stride)
    elif isinstance(module, torch.nn.Linear):
        num_feature = module.in_features
        handle = module.register_forward_hook(_linear_statistics_hook)
    else:
        raise TypeError

    module.register_buffer(
        'sample_counter', torch.zeros(1, dtype=torch.int))
    module.register_buffer(
        'mu', torch.zeros(num_feature, dtype=torch.float64))
    module.register_buffer(
        'sigma', torch.zeros((num_feature, num_feature), dtype=torch.float64))

    return handle


def _conv2d_statistics_hook(module, X, _):
    X = X[0].clone().detach().to(torch.float64)
    batch_size, in_channels, X_with, X_height = X.shape
    assert in_channels == module.in_channels
    num_feature = in_channels * module.kernel_size[0] * module.kernel_size[1]

    X = unfold(
            X,
            kernel_size=module.kernel_size,
            padding=module.padding,
            stride=(
                module.stride[0] * module.sampling_stride,
                module.stride[1] * module.sampling_stride
            ),
        )
    assert X.shape[0] == batch_size
    assert X.shape[1] == num_feature
    unfold_size = X.shape[2]
    new_count = batch_size * unfold_size

    X = X.transpose(1, 2)
    X = X.reshape((-1, num_feature))

    mu = X.mean(axis=0).to(module.mu.device)
    assert mu.shape == (num_feature,)

    sigma = torch.matmul(X.T, X)
    sigma = sigma / new_count

    old_count = int(module.sample_counter)

    module.mu = \
        module.mu * (old_count/(old_count+new_count)) + \
        mu * (new_count/(old_count+new_count))

    module.sigma = \
        module.sigma * (old_count/(old_count+new_count)) + \
        sigma * (new_count/(old_count+new_count))

    module.sample_counter += new_count


def _linear_statistics_hook(module, X, _):
    X = X[0].clone().detach().to(torch.float64)
    new_count, num_feature = X.shape
    assert num_feature == module.in_features

    mu = X.mean(axis=0).to(module.mu.device)
    assert mu.shape == (num_feature, )

    # often too memory-hungery
    # sigma = torch.matmul(X[:, :, None], X[:, None, :])
    # sigma = sigma.mean(axis=0)
    # torch.cuda.empty_cache()
    sigma = torch.matmul(X.T, X)
    sigma = sigma / new_count

    old_count = int(module.sample_counter)

    module.sigma = \
        module.sigma * (old_count/(old_count+new_count)) + \
        sigma * (new_count/(old_count+new_count))

    module.mu = \
        module.mu * (old_count/(old_count+new_count)) + \
        mu * (new_count/(old_count+new_count))

    module.sample_counter += new_count
