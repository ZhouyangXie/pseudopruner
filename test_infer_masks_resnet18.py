import torch
from torchvision.models import resnet18

from pseudopruner.channel_pruner import RandomChannelPruner
from pseudopruner.infer_masks import infer_masks
from pseudopruner.utils import \
    get_ready_to_prune, mark_to_prune


def zero_weight(model, mask_weight):
    """
    Make the 'prune_channel_mask' and 'prune_weight_mask'
    effective by assigning 0 to specific parameters.

    """
    with torch.no_grad():
        for module in model.modules():
            if mask_weight and hasattr(module, 'prune_weight_mask'):
                assert isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
                module.weight[module.prune_weight_mask, ] = 0
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias[module.prune_weight_mask, ] = 0

            if hasattr(module, 'prune_channel_mask'):
                module.weight[:, module.prune_channel_mask, ] = 0


def test():
    model = resnet18()
    model.eval()
    device = 'cuda:0'

    model = model.to(device)
    dummy_input = torch.rand((1, 3, 256, 256)).to(device)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if 'layer3' in name or 'layer1' in name:
                mark_to_prune(module, {'sparsity': 0.4})

    # prune the model by masking
    get_ready_to_prune(model)
    RandomChannelPruner().compute_mask(model)
    # model.layer4[-1].conv1.prune_channel_mask[0] = True

    # infer other masks
    infer_masks(model, dummy_input)

    # make masked weights zero
    zero_weight(model, False)

    with torch.no_grad():
        y0 = model(dummy_input).detach().clone()

    zero_weight(model, True)

    with torch.no_grad():
        y1 = model(dummy_input).detach().clone()

    assert (y0 == y1).all()


if __name__ == '__main__':
    test()
