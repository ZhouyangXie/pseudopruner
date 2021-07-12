import torch
from torchvision.models import resnet18

from pseudopruner.weight_pruner import NormPruner
from pseudopruner.utils import get_ready_to_prune, mark_to_prune, \
    count_flops, infer_masks


def test():
    model = resnet18()
    model.eval()
    device = 'cuda:0'

    model = model.to(device)
    dummy_input = 10 * torch.rand((1, 3, 256, 256)).to(device)
    flops_before = count_flops(model, dummy_input)
    with torch.no_grad():
        y_before = model(dummy_input).detach().clone()

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            mark_to_prune(module, {'sparsity': 0.0})

    # prune the model by masking
    get_ready_to_prune(model)
    pruner = NormPruner()
    pruner.compute_mask(model)
    # infer other masks
    infer_masks(model, dummy_input)

    flops_after = count_flops(model, dummy_input)
    with torch.no_grad():
        y_after = model(dummy_input).detach().clone()

    assert all([
        (~mod.prune_channel_mask).all()
        for mod in model.modules() if hasattr(mod, 'prune_channel_mask')
    ])
    assert all([
        (~mod.prune_weight_mask).all()
        for mod in model.modules() if hasattr(mod, 'prune_weight_mask')
    ])
    assert flops_after == flops_before
    assert (y_before == y_after).all()


if __name__ == '__main__':
    test()
