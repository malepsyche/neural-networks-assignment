from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from captum.attr import LayerGradCam

from b1_model import AlexNetLike
from b2_model import VGGLike
from b3_model import ResNetLike


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "chestxray_binary_64.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "b4"
)

B1_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "b1"
    / "best_model.pt"
)

B2_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "b2"
    / "best_model.pt"
)

B3_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "b3"
    / "best_model.pt"
)

SEED = 42
NUM_IMAGES = 5

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def load_test_data():
    data = np.load(DATA_PATH)

    raw_images = data[
        "test_images"
    ].astype(np.float32)

    labels = data[
        "test_labels"
    ].astype(np.int64)

    # need the training data because b1-b3 normalised all images using training statistics
    train_images = data[
        "train_images"
    ].astype(np.float32)

    train_images = (
        train_images / 255.0
    )

    test_images = (
        raw_images / 255.0
    )

    train_mean = (
        train_images.mean()
    )

    train_std = (
        train_images.std()
    )

    test_normalized = (
        test_images - train_mean
    ) / train_std

    test_normalized = torch.tensor(
        test_normalized,
        dtype=torch.float32,
    )

    if test_normalized.ndim == 3:
        test_normalized = (
            test_normalized.unsqueeze(1)
        )

    return (
        raw_images,
        labels,
        test_normalized,
    )


def select_pneumonia_images(
    labels,
):
    # assumes label 1 = pneumonia, consistent with the sensitivity calculations in b1-b3
    pneumonia_indices = np.where(
        labels == 1
    )[0]

    rng = np.random.default_rng(
        SEED
    )

    selected_indices = rng.choice(
        pneumonia_indices,
        size=NUM_IMAGES,
        replace=False,
    )

    return selected_indices


def load_models():
    b1 = AlexNetLike().to(DEVICE)

    b1.load_state_dict(
        torch.load(
            B1_MODEL_PATH,
            map_location=DEVICE,
        )
    )

    b2 = VGGLike().to(DEVICE)

    b2.load_state_dict(
        torch.load(
            B2_MODEL_PATH,
            map_location=DEVICE,
        )
    )

    b3 = ResNetLike().to(DEVICE)

    b3.load_state_dict(
        torch.load(
            B3_MODEL_PATH,
            map_location=DEVICE,
        )
    )

    b1.eval()
    b2.eval()
    b3.eval()

    return {
        "B1 AlexNet-like": b1,
        "B2 VGG-like": b2,
        "B3 ResNet-like": b3,
    }


def get_target_layers(
    b1,
    b2,
    b3,
):
    """
    grad-cAM is normally applied to the final
    convolutional feature layer

    these correspond to the last convolutional
    layers before pooling/classification
    """

    return {
        # last conv2d in b1 features
        "B1 AlexNet-like":
            b1.features[8],

        # last conv2d in b2 features
        "B2 VGG-like":
            b2.features[17],

        # second convolution of final residual block
        "B3 ResNet-like":
            b3.block3.conv2,
    }


def compute_gradcam(
    model,
    target_layer,
    image,
):

    image = (
        image
        .unsqueeze(0)
        .to(DEVICE)
    )

    image.requires_grad_(True)

    gradcam = LayerGradCam(
        model,
        target_layer,
    )

    attribution = gradcam.attribute(
        image
    )

    # upsample gradcam map to original 64x64.
    attribution = F.interpolate(
        attribution,
        size=(64, 64),
        mode="bilinear",
        align_corners=False,
    )

    # positive evidence for pneumonia.
    attribution = torch.relu(
        attribution
    )

    heatmap = (
        attribution
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    # normalize to [0, 1] to visualise
    minimum = heatmap.min()
    maximum = heatmap.max()

    if maximum > minimum:
        heatmap = (
            heatmap - minimum
        ) / (
            maximum - minimum
        )

    return heatmap


@torch.no_grad()
def get_probability(
    model,
    image,
):
    image = (
        image
        .unsqueeze(0)
        .to(DEVICE)
    )

    logit = model(image)

    probability = torch.sigmoid(
        logit
    )

    return probability.item()


def plot_comparison(
    raw_images,
    normalized_images,
    selected_indices,
    models,
    target_layers,
):
    model_names = list(
        models.keys()
    )

    # 5 rows: one row for each selected pneumonia image
    # 4 columns: original + b1 + b2 + b3
    fig, axes = plt.subplots(
        NUM_IMAGES,
        4,
        figsize=(14, 16),
    )

    for row, image_index in enumerate(
        selected_indices
    ):
        raw_image = raw_images[
            image_index
        ]

        normalized_image = (
            normalized_images[
                image_index
            ]
        )

        # original X-ray
        axes[row, 0].imshow(
            raw_image,
            cmap="gray",
        )

        axes[row, 0].set_title(
            f"X-ray {image_index}"
        )

        axes[row, 0].axis("off")

        # Grad-CAM for each architecture
        for column, model_name in enumerate(
            model_names,
            start=1,
        ):
            model = models[
                model_name
            ]

            target_layer = (
                target_layers[
                    model_name
                ]
            )

            heatmap = compute_gradcam(
                model,
                target_layer,
                normalized_image,
            )

            probability = (
                get_probability(
                    model,
                    normalized_image,
                )
            )

            axes[row, column].imshow(
                raw_image,
                cmap="gray",
            )

            axes[row, column].imshow(
                heatmap,
                cmap="jet",
                alpha=0.45,
            )

            axes[row, column].set_title(
                f"{model_name}\n"
                f"P(pneumonia)="
                f"{probability:.3f}"
            )

            axes[row, column].axis(
                "off"
            )

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "gradcam_comparison.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print(
        f"Wrote {path}"
    )


def main():
    print(
        f"Using device: {DEVICE}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        raw_images,
        labels,
        normalized_images,
    ) = load_test_data()

    print(
        "Test label counts:",
        dict(
            zip(
                *np.unique(
                    labels,
                    return_counts=True,
                )
            )
        ),
    )

    selected_indices = (
        select_pneumonia_images(
            labels
        )
    )

    print(
        "Selected pneumonia indices:",
        selected_indices,
    )

    models = load_models()

    target_layers = get_target_layers(
        models["B1 AlexNet-like"],
        models["B2 VGG-like"],
        models["B3 ResNet-like"],
    )

    print("\nTarget layers:")

    for name, layer in (
        target_layers.items()
    ):
        print(
            f"{name}: {layer}"
        )

    plot_comparison(
        raw_images,
        normalized_images,
        selected_indices,
        models,
        target_layers,
    )


if __name__ == "__main__":
    main()