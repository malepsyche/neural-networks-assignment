import copy
from pathlib import Path

import numpy as np
import torch

from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from b3_model import ResNetLike


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "chestxray_binary_64.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "b3"
)

# best hyperparameters from b1/b2
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

MAX_EPOCHS = 50
SEED = 42

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def load_data(path):
    data = np.load(path)

    print(
        "Dataset keys:",
        data.files,
    )

    x_train = torch.tensor(
        data["train_images"],
        dtype=torch.float32,
    )

    y_train = torch.tensor(
        data["train_labels"],
        dtype=torch.float32,
    )

    x_validation = torch.tensor(
        data["val_images"],
        dtype=torch.float32,
    )

    y_validation = torch.tensor(
        data["val_labels"],
        dtype=torch.float32,
    )

    x_test = torch.tensor(
        data["test_images"],
        dtype=torch.float32,
    )

    y_test = torch.tensor(
        data["test_labels"],
        dtype=torch.float32,
    )

    # scale to [0, 1]
    x_train = x_train / 255.0
    x_validation = x_validation / 255.0
    x_test = x_test / 255.0

    train_mean = x_train.mean()

    train_std = (
        x_train
        .std()
        .clamp_min(1e-8)
    )

    x_train = (
        x_train - train_mean
    ) / train_std

    x_validation = (
        x_validation - train_mean
    ) / train_std

    x_test = (
        x_test - train_mean
    ) / train_std

    # N,H,W -> N,C,H,W
    if x_train.ndim == 3:
        x_train = x_train.unsqueeze(1)

        x_validation = (
            x_validation.unsqueeze(1)
        )

        x_test = x_test.unsqueeze(1)

    print(
        f"Training mean: "
        f"{train_mean.item():.6f}"
    )

    print(
        f"Training std: "
        f"{train_std.item():.6f}"
    )

    print(
        f"Train shape: "
        f"{tuple(x_train.shape)}"
    )

    print(
        f"Validation shape: "
        f"{tuple(x_validation.shape)}"
    )

    print(
        f"Test shape: "
        f"{tuple(x_test.shape)}"
    )

    return {
        "train": TensorDataset(
            x_train,
            y_train,
        ),

        "validation": TensorDataset(
            x_validation,
            y_validation,
        ),

        "test": TensorDataset(
            x_test,
            y_test,
        ),
    }


@torch.no_grad()
def evaluate(model, dataset):
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    loss_function = (
        nn.BCEWithLogitsLoss()
    )

    total_loss = 0.0
    total_samples = 0

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for images, targets in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        logits = model(images)

        loss = loss_function(
            logits,
            targets,
        )

        current_batch_size = len(
            targets
        )

        total_loss += (
            loss.item()
            * current_batch_size
        )

        total_samples += (
            current_batch_size
        )

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= 0.5
        ).long()

        targets_int = targets.long()

        true_positive += (
            (predictions == 1)
            & (targets_int == 1)
        ).sum().item()

        true_negative += (
            (predictions == 0)
            & (targets_int == 0)
        ).sum().item()

        false_positive += (
            (predictions == 1)
            & (targets_int == 0)
        ).sum().item()

        false_negative += (
            (predictions == 0)
            & (targets_int == 1)
        ).sum().item()

    accuracy = (
        true_positive
        + true_negative
    ) / total_samples

    sensitivity = (
        true_positive
        / (
            true_positive
            + false_negative
        )
        if (
            true_positive
            + false_negative
        ) > 0
        else 0.0
    )

    specificity = (
        true_negative
        / (
            true_negative
            + false_positive
        )
        if (
            true_negative
            + false_positive
        ) > 0
        else 0.0
    )

    return {
        "loss":
            total_loss / total_samples,

        "accuracy":
            accuracy,

        "error":
            1.0 - accuracy,

        "sensitivity":
            sensitivity,

        "specificity":
            specificity,
    }


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def train_model(
    model,
    train_data,
    validation_data,
):
    generator = torch.Generator()

    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    loss_function = (
        nn.BCEWithLogitsLoss()
    )

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_error = float("inf")
    best_loss = float("inf")

    best_epoch = 0
    best_metrics = None
    best_weights = None

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        model.train()

        for images, targets in train_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimiser.zero_grad()

            logits = model(images)

            loss = loss_function(
                logits,
                targets,
            )

            loss.backward()
            optimiser.step()

        metrics = evaluate(
            model,
            validation_data,
        )

        improved = (
            metrics["error"] < best_error
            or (
                metrics["error"]
                == best_error
                and metrics["loss"]
                < best_loss
            )
        )

        if improved:
            best_error = (
                metrics["error"]
            )

            best_loss = (
                metrics["loss"]
            )

            best_epoch = epoch

            best_metrics = metrics

            best_weights = copy.deepcopy(
                model.state_dict()
            )

        print(
            f"epoch {epoch:02d} | "
            f"val loss "
            f"{metrics['loss']:.4f} | "
            f"val accuracy "
            f"{metrics['accuracy']:.4f}"
        )

    model.load_state_dict(
        best_weights
    )

    return (
        best_epoch,
        best_metrics,
        best_weights,
    )


def main():
    print(
        f"Using device: {DEVICE}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE:.0e}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.manual_seed(SEED)

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )

    data = load_data(
        DATA_PATH
    )

    model = ResNetLike().to(
        DEVICE
    )

    parameter_count = (
        count_parameters(model)
    )

    print(
        "\n=============================="
    )

    print(
        f"Learnable parameters: "
        f"{parameter_count:,}"
    )

    print(
        "=============================="
    )

    (
        best_epoch,
        best_metrics,
        best_weights,
    ) = train_model(
        model,
        data["train"],
        data["validation"],
    )

    print(
        "\n=============================="
    )

    print("BEST VALIDATION MODEL")

    print(
        "=============================="
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Validation loss: "
        f"{best_metrics['loss']:.4f}"
    )

    print(
        f"Validation accuracy: "
        f"{best_metrics['accuracy']:.4f}"
    )

    print(
        f"Validation classification error: "
        f"{best_metrics['error']:.4f}"
    )

    model.load_state_dict(
        best_weights
    )

    test_metrics = evaluate(
        model,
        data["test"],
    )

    torch.save(
        model.state_dict(),
        OUTPUT_DIR / "best_model.pt",
    )

    print(
        "\n=============================="
    )

    print("FINAL TEST RESULTS")

    print(
        "=============================="
    )

    print(
        f"Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Sensitivity: "
        f"{test_metrics['sensitivity']:.4f}"
    )

    print(
        f"Specificity: "
        f"{test_metrics['specificity']:.4f}"
    )

    print(
        f"Classification error: "
        f"{test_metrics['error']:.4f}"
    )

    print(
        f"Learnable parameters: "
        f"{parameter_count:,}"
    )


if __name__ == "__main__":
    main()