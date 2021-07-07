import torch
from contextlib import contextmanager


_allowed_prune_layers = (
    torch.nn.Conv2d,
    torch.nn.Linear
)


def get_ready_to_prune(model):
    '''
    Called by the user to get the model ready for pruning.
    It will register some attributes to some layers that are
    agonstic to normal training/inference behaviours.

    Args:
        module (torch.Module): the module to be pruned.
    '''
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            module.register_buffer(
                'prune_channel_mask',
                torch.zeros(module.in_channels, dtype=torch.bool)
            )
            module.register_buffer(
                'prune_weight_mask',
                torch.zeros(module.out_channels, dtype=torch.bool)
            )
        elif isinstance(module, torch.nn.Linear):
            module.register_buffer(
                'prune_channel_mask',
                torch.zeros(module.in_features, dtype=torch.bool)
            )
            module.register_buffer(
                'prune_weight_mask',
                torch.zeros(module.out_features, dtype=torch.bool)
            )
        else:
            continue


def mark_to_prune(module, kwargs):
    """
    Called by the user to mark a Conv2d or Linear module as 'to-prune'

    Args:
        module (torch.Module): the module to be pruned.
        kwargs (dict): other necessary arguments passed to pruners
    """
    assert isinstance(module, _allowed_prune_layers)

    setattr(module, 'to_prune', True)
    for k, v in kwargs.items():
        setattr(module, k, v)


def _hook_set_input_zero(module, inputs):
    assert isinstance(module, _allowed_prune_layers)
    assert hasattr(module, 'prune_channel_mask')
    assert len(inputs) == 1
    x_handled = inputs[0].clone()
    x_handled[:, module.prune_channel_mask] = 0
    return (x_handled, )


@contextmanager
def make_pruning_effective(model):
    """
    Make the 'prune_channel_mask' and 'prune_weight_mask'
    effective by setting module input to zero accordingly
    inside the context.

    It is assumed that the channel/weight masks are all
    inferred, so there's no need to mask weights or outputs.
    """
    handles = []
    for module in model.modules():
        if hasattr(module, 'prune_channel_mask'):
            h = module.register_forward_pre_hook(_hook_set_input_zero)
            handles.append(h)

    try:
        yield model
    finally:
        for h in handles:
            h.remove()
