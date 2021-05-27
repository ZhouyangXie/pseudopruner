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
        v = grad_output.abs().sum(axis=0)
        additional_weight_mask = ~(torch.isnan(v) | (v == 0))
    elif grad_output.dim() == 4:
        v = grad_output.abs().sum(axis=(0, 2, 3))
        additional_weight_mask = ~(torch.isnan(v) | (v == 0))
    else:
        raise RuntimeError('wrong output grad dim')

    assert hasattr(module, 'prune_weight_mask')
    module.prune_weight_mask[additional_weight_mask, ] = True

    grad_input_handled = grad_input.clone()
    assert hasattr(module, 'prune_channel_mask')
    grad_input_handled[:, ~module.prune_channel_mask] = float('nan')
    grad_input_handled[:, module.prune_channel_mask] = 1

    return (grad_input_handled, )


def infer_masks(model, dummy_input):
    model = model.eval()

    # get the channel-merging layers ready to handle NaN
    nan_handlers = []
    for module in model.modules():
        # TODO: to add more channel-merging layers for better compatibility
        if isinstance(module, _allowed_prune_layers):
            h = module.register_forward_pre_hook(_hook_handle_nan_input)
            nan_handlers.append(h)

    # forward-propogate the network to make some input channels NaN
    with torch.no_grad():
        # set all masked weights to NaN
        for module in model.modules():
            if hasattr(module, 'prune_weight_mask'):
                assert isinstance(
                    module, (torch.nn.Conv2d, torch.nn.Linear))
                module.weight[module.prune_weight_mask, ] = float('nan')

        y = model(dummy_input)
        assert not torch.isnan(y).any(), \
            'should not prune the weight of the last conv/linear layer'

        # set all masked NaN weights to zero
        for module in model.modules():
            if hasattr(module, 'prune_weight_mask'):
                module.weight[module.prune_weight_mask, ] = 0

    for h in nan_handlers:
        h.remove()

    # get the channel-merging layers(Conv2d and Linear here)
    # ready to handle NaN in the gradients:
    nan_handlers = []
    for module in model.modules():
        # TODO: to add more channel-merging layers for better compatibility
        if isinstance(module, _allowed_prune_layers):
            h = module.register_full_backward_hook(
                _hook_handle_nan_output_grad)
            nan_handlers.append(h)

    with torch.enable_grad():
        # to enable the backward on the first layer
        dummy_input.requires_grad = True
        y = model(dummy_input)
        loss = y.sum()
        loss = loss * torch.sqrt(-torch.ones(loss.shape))
        loss.backward()

    for h in nan_handlers:
        h.remove()
