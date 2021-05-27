import torch
from .utils import _allowed_prune_layers


def _hook_handle_nan_input(module, inputs):
    '''
        this hook identify all NaN channels in the input,
        set some more channel masks,
        and turn those inputs to 0 before forwarding
    '''
    assert isinstance(module, _allowed_prune_layers)
    assert len(inputs) == 1
    x = inputs[0]

    if x.dim() == 2:
        additional_channel_mask = torch.isnan(x[0, :])
    elif x.dim() == 4:
        additional_channel_mask = torch.isnan(x[0, :, 0, 0])
    else:
        raise RuntimeError('wrong input dim')

    assert hasattr(module, 'prune_channel_mask')
    module.prune_channel_mask[additional_channel_mask, ] = True

    x_handled = x.clone()
    x_handled[torch.isnan(x)] = 1
    return (x_handled, )


def _hook_handle_nan_output_grad(module, grad_inputs, grad_outputs):
    '''
        this hook identify all 0 channels in the output,
        set some more weight masks,
        and turn the gradient to 0 before continuing the backward
    '''
    assert isinstance(module, _allowed_prune_layers)
    assert len(grad_inputs) == 1
    assert len(grad_outputs) == 1
    grad_input = grad_inputs[0]
    grad_output = grad_outputs[0]

    if grad_output.dim() == 2:
        additional_weight_mask = (grad_output == 0).all(axis=(0))
    elif grad_output.dim() == 4:
        additional_weight_mask = (
            grad_output == 0
        ).all(axis=0).all(axis=1).all(axis=1)
    else:
        raise RuntimeError('wrong output grad dim')

    assert hasattr(module, 'prune_weight_mask')
    module.prune_weight_mask[additional_weight_mask, ] = True

    grad_input_handled = grad_input.clone()
    assert hasattr(module, 'prune_channel_mask')
    grad_input_handled[:, module.prune_channel_mask] = 0
    # grad_input_handled[:, ~module.prune_channel_mask] = 1

    return (grad_input_handled, )


def infer_masks(model, dummy_input):
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # set all masked weights to NaN
    for module in model.modules():
        if hasattr(module, 'prune_weight_mask'):
            assert isinstance(
                module, (torch.nn.Conv2d, torch.nn.Linear))
            module.weight[module.prune_weight_mask, ] = float('nan')

    # get the channel-merging layers(Conv2d and Linear here)
    # ready to handle NaN: set additional channel mask and turn NaN to zero
    nan_handlers = []
    for module in model.modules():
        # TODO: to add more channel-merging layers for better compatibility
        if isinstance(module, _allowed_prune_layers):
            h = module.register_forward_pre_hook(_hook_handle_nan_input)
            nan_handlers.append(h)

    # forward-propogate the network to make some input channels NaN
    y = model(dummy_input)
    for h in nan_handlers:
        h.remove()

    # set all masked NaN weights to zero
    # to infer additional weight mask
    for module in model.modules():
        if hasattr(module, 'prune_weight_mask'):
            module.weight[module.prune_weight_mask, ] = 1

    model.train()
    for param in model.parameters():
        param.requires_grad = True

    # get the channel-merging layers(Conv2d and Linear here)
    # ready to handle NaN in the gradients:
    #   set channel masked gradients to zero
    #   set input gradient who's channel-masked to zero, others to one
    nan_handlers = []
    for module in model.modules():
        # TODO: to add more channel-merging layers for better compatibility
        if isinstance(module, _allowed_prune_layers):
            h = module.register_full_backward_hook(
                _hook_handle_nan_output_grad)
            nan_handlers.append(h)

    dummy_input.requires_grad = True
    y = model(dummy_input)
    loss = y.sum()
    assert not torch.isnan(loss).any(), \
        'should not prune the weight of the last conv/linear layer'
    loss.backward()
    for h in nan_handlers:
        h.remove()

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
