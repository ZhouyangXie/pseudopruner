import logging
import torch

_logger = logging.getLogger(__name__)

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


def make_pruning_effective(model):
    """
    Make the 'prune_channel_mask' and 'prune_weight_mask'
    effective by assigning 0 to specific parameters.

    """
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, 'prune_weight_mask'):
                assert isinstance(module, _allowed_prune_layers)
                module.weight[module.prune_weight_mask, ] = 0
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias[module.prune_weight_mask, ] = 0

            if hasattr(module, 'prune_channel_mask'):
                module.weight[:, module.prune_channel_mask, ] = 0
