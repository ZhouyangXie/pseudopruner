import torch
from .utils import _allowed_prune_layers


def _hook_count_conv2d_flops(m, x, y):
    assert isinstance(m, torch.nn.Conv2d)
    assert len(x) == 1

    _, c_in, _, _ = x[0].shape
    assert c_in == m.in_channels
    if hasattr(m, 'prune_channel_mask'):
        c_in = c_in - m.prune_channel_mask.sum()

    _, c_out, w, h = y.shape
    assert c_out == m.out_channels
    if hasattr(m, 'prune_weight_mask'):
        c_out = c_out - m.prune_weight_mask.sum()

    kw, kh = m.kernel_size
    b = 0 if m.bias is None else 1

    flops = c_out * w * h * (c_in * kw * kh + b)
    setattr(m, 'flops', flops)


def _hook_count_linear_flops(m, x, y):
    assert isinstance(m, torch.nn.Linear)
    assert len(x) == 1

    _, c_in = x[0].shape
    assert c_in == m.in_features
    if hasattr(m, 'prune_channel_mask'):
        c_in = c_in - m.prune_channel_mask.sum()

    _, c_out = y.shape
    assert c_out == m.out_features
    if hasattr(m, 'prune_weight_mask'):
        c_out = c_out - m.prune_weight_mask.sum()

    b = 0 if m.bias is None else 1

    flops = c_out * (c_in + b)
    setattr(m, 'flops', flops)


def count_flops(model, dummy_input):
    """
        Compute the FLOPS of model inference.
        The pruned channels/weights are excluded.
        Only Conv2d/Linear layers count.

    Args:
        model (torch.nn.Module): model already stored on a proper device
        dummy_input (torch.Tensor):
            input of proper size on same device as the model

    Returns:
        flops (int): total number of FLOPS
    """
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    handles = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            h = module.register_forward_hook(_hook_count_conv2d_flops)
        elif isinstance(module, torch.nn.Linear):
            h = module.register_forward_hook(_hook_count_linear_flops)
        else:
            continue
        handles.append(h)

    _ = model(dummy_input)

    for h in handles:
        h.remove()

    flops = 0
    for module in model.modules():
        if isinstance(module, _allowed_prune_layers):
            flops += module.flops

    return flops