import torch
from torchvision.models import resnet18

from pseudopruner.weight_pruner import NormPruner
from pseudopruner.count_flops import count_flops
from pseudopruner.infer_masks import infer_masks
from pseudopruner.utils import \
    get_ready_to_prune, mark_to_prune, make_pruning_effective


def main():
    device = 'cuda:1'
    model = resnet18().to(device)
    dummy_input = torch.rand((1, 3, 224, 224)).to(device)

    full_flops = count_flops(model, dummy_input)
    print(f'resnet18 FLOPS: {full_flops}')

    # register buffer for masks
    get_ready_to_prune(model)

    # mark some convs as 'to-prune'
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if 'layer1' in name or 'layer3' in name:
                mark_to_prune(
                    module,
                    {'sparsity': 0.5, 'norm': 2}
                )

    # update the weight masks
    pruner = NormPruner()
    pruner.compute_mask(model)

    # infer other channel/weight masks
    infer_masks(model, dummy_input)

    with make_pruning_effective(model) as pruned_model:
        _ = pruned_model(dummy_input)
        # do anything

    # skip the finetuning or compensation step
    partial_flops = count_flops(model, dummy_input)
    print(f'pruned resnet18 FLOPS: {partial_flops}')


if __name__ == "__main__":
    main()
