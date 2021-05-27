import torch

from pseudopruner.utils import get_ready_to_prune
from pseudopruner.infer_masks import infer_masks


class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, bias=True)
        self.conv2 = torch.nn.Conv2d(4, 8, 3, bias=True)
        self.conv3 = torch.nn.Conv2d(4, 8, 3, bias=False)
        self.conv4 = torch.nn.Conv2d(8, 4, 3, bias=True)
        self.conv5 = torch.nn.Conv2d(4, 1, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x0 = self.conv2(x)
        x1 = self.conv3(x)
        x2 = x0 + x1
        x3 = self.conv4(x2)
        x4 = self.conv5(x3)
        return torch.square(x4).sum()


def test():
    model = SimpleNet()
    get_ready_to_prune(model)

    model.conv1.prune_weight_mask = torch.tensor(
        [0, 0, 1, 0], dtype=torch.bool)
    model.conv2.prune_channel_mask = torch.tensor(
        [0, 1, 0, 0], dtype=torch.bool)
    model.conv3.prune_channel_mask = torch.tensor(
        [0, 0, 1, 0], dtype=torch.bool)

    model.conv2.prune_weight_mask = torch.tensor(
        [1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool)
    model.conv3.prune_weight_mask = torch.tensor(
        [1, 0, 1, 0, 0, 0, 0, 0], dtype=torch.bool)
    model.conv4.prune_channel_mask = torch.tensor(
        [1, 0, 0, 1, 0, 0, 0, 0], dtype=torch.bool
    )

    model.conv4.prune_weight_mask = torch.tensor(
        [1, 1, 0, 0], dtype=torch.bool)
    model.conv5.prune_channel_mask = torch.tensor(
        [1, 0, 1, 0], dtype=torch.bool)

    dummy_input = 10 * torch.rand((1, 3, 11, 11))
    infer_masks(model, dummy_input)

    assert (model.conv1.prune_channel_mask == torch.tensor(
        [0, 0, 0], dtype=torch.bool)).all()
    assert (model.conv1.prune_weight_mask == torch.tensor(
        [0, 0, 1, 0], dtype=torch.bool)).all()

    assert (model.conv2.prune_channel_mask == torch.tensor(
        [0, 1, 1, 0], dtype=torch.bool)).all()
    assert (model.conv2.prune_weight_mask == torch.tensor(
        [1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool)).all()

    assert (model.conv3.prune_channel_mask == torch.tensor(
        [0, 0, 1, 0], dtype=torch.bool)).all()
    assert (model.conv3.prune_weight_mask == torch.tensor(
        [1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool)).all()

    assert (model.conv4.prune_channel_mask == torch.tensor(
        [1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool)).all()
    assert (model.conv4.prune_weight_mask == torch.tensor(
        [1, 1, 1, 0], dtype=torch.bool)).all()

    assert (model.conv5.prune_channel_mask == torch.tensor(
        [1, 1, 1, 0], dtype=torch.bool)).all()
    assert (model.conv5.prune_weight_mask == torch.tensor(
        [0, ], dtype=torch.bool)).all()


if __name__ == "__main__":
    test()
