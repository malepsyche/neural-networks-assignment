import torch
from thop import profile

from b1_model import AlexNetLike
from b2_model import VGGLike
from b3_model import ResNetLike


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def get_complexity(model):
    model = model.cpu()
    model.eval()

    dummy_input = torch.randn(
        1,
        1,
        64,
        64,
    )

    macs, _ = profile(
        model,
        inputs=(dummy_input,),
        verbose=False,
    )

    flops = 2 * macs

    return int(macs), int(flops)


def main():
    models = {
        "B1 AlexNet-like": AlexNetLike(),
        "B2 VGG-like": VGGLike(),
        "B3 ResNet-like": ResNetLike(),
    }

    print(
        f"{'Architecture':<20}"
        f"{'Parameters':>15}"
        f"{'MACs':>15}"
        f"{'FLOPs':>15}"
    )

    print("-" * 65)

    for name, model in models.items():
        parameters = count_parameters(
            model
        )

        macs, flops = get_complexity(
            model
        )

        print(
            f"{name:<20}"
            f"{parameters:>15,}"
            f"{macs:>15,}"
            f"{flops:>15,}"
        )


if __name__ == "__main__":
    main()