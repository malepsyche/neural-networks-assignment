import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from b2_model import VGGLike


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "chestxray_binary_64.npz"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "b2"

LEARNING_RATE = 1e-3
MAX_EPOCHS = 50
SEED = 42

BATCH_SIZES = [
    16,
    32,
    64,
    128,
    256,
]

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def load_data(path):
    data = np.load(path)

    print("Dataset keys:", data.files)

    x_train = data["train_images"]
    y_train = data["train_labels"]

    x_validation = data["val_images"]
    y_validation = data["val_labels"]

    x_test = data["test_images"]
    y_test = data["test_labels"]

    x_train = torch.tensor(
        x_train,
        dtype=torch.float32,
    )

    x_validation = torch.tensor(
        x_validation,
        dtype=torch.float32,
    )

    x_test = torch.tensor(
        x_test,
        dtype=torch.float32,
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.float32,
    )

    y_validation = torch.tensor(
        y_validation,
        dtype=torch.float32,
    )

    y_test = torch.tensor(
        y_test,
        dtype=torch.float32,
    )

    # Scale pixel intensities to [0, 1]
    x_train = x_train / 255.0
    x_validation = x_validation / 255.0
    x_test = x_test / 255.0

    # Compute normalization statistics using training set 
    train_mean = x_train.mean()
    train_std = x_train.std().clamp_min(1e-8)

    x_train = (
        x_train - train_mean
    ) / train_std

    x_validation = (
        x_validation - train_mean
    ) / train_std

    x_test = (
        x_test - train_mean
    ) / train_std

    # [N, H, W] -> [N, 1, H, W]
    if x_train.ndim == 3:
        x_train = x_train.unsqueeze(1)
        x_validation = x_validation.unsqueeze(1)
        x_test = x_test.unsqueeze(1)

    print(
        f"Training mean: {train_mean.item():.6f}"
    )

    print(
        f"Training std: {train_std.item():.6f}"
    )

    print(
        f"Train shape: {tuple(x_train.shape)}"
    )

    print(
        f"Validation shape: "
        f"{tuple(x_validation.shape)}"
    )

    print(
        f"Test shape: {tuple(x_test.shape)}"
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
def evaluate(
    model,
    dataset,
    batch_size,
):
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    loss_function = nn.BCEWithLogitsLoss()

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

        current_batch_size = len(targets)

        total_loss += (
            loss.item() * current_batch_size
        )

        total_samples += current_batch_size

        probabilities = torch.sigmoid(logits)

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
        true_positive + true_negative
    ) / total_samples

    sensitivity = (
        true_positive
        / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    specificity = (
        true_negative
        / (true_negative + false_positive)
        if true_negative + false_positive > 0
        else 0.0
    )

    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy,
        "error": 1.0 - accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def train_model(
    model,
    train_data,
    validation_data,
    batch_size,
):
    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    loss_function = nn.BCEWithLogitsLoss()

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = []

    best_error = float("inf")
    best_loss = float("inf")
    best_epoch = 0
    best_weights = None
    best_metrics = None

    for epoch in range(1, MAX_EPOCHS + 1):
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
            batch_size,
        )

        history.append(
            {
                "epoch": epoch,
                "validation_loss": metrics["loss"],
                "validation_accuracy": metrics["accuracy"],
            }
        )

        improved = (
            metrics["error"] < best_error
            or (
                metrics["error"] == best_error
                and metrics["loss"] < best_loss
            )
        )

        if improved:
            best_error = metrics["error"]
            best_loss = metrics["loss"]
            best_epoch = epoch
            best_metrics = metrics

            best_weights = copy.deepcopy(
                model.state_dict()
            )

        print(
            f"Batch {batch_size:3d} | "
            f"epoch {epoch:02d} | "
            f"val loss {metrics['loss']:.4f} | "
            f"val accuracy {metrics['accuracy']:.4f}"
        )

    model.load_state_dict(best_weights)

    return (
        pd.DataFrame(history),
        best_epoch,
        best_metrics,
        best_weights,
    )


def plot_validation_loss(histories):
    plt.figure(figsize=(8, 5))

    for batch_size, history in histories.items():
        plt.plot(
            history["epoch"],
            history["validation_loss"],
            label=f"Batch = {batch_size}",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title(
        "Validation Loss for Different Batch Sizes"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "validation_loss.png"
    )

    plt.savefig(
        path,
        dpi=300,
    )

    plt.show()

    print(f"Wrote {path}")


def plot_validation_accuracy(histories):
    plt.figure(figsize=(8, 5))

    for batch_size, history in histories.items():
        plt.plot(
            history["epoch"],
            history["validation_accuracy"],
            label=f"Batch = {batch_size}",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(
        "Validation Accuracy for Different Batch Sizes"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "validation_accuracy.png"
    )

    plt.savefig(
        path,
        dpi=300,
    )

    plt.show()

    print(f"Wrote {path}")


def main():
    print(f"Using device: {DEVICE}")
    print(
        f"Learning rate from B1: "
        f"{LEARNING_RATE:.0e}"
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

    data = load_data(DATA_PATH)

    histories = {}
    results = {}
    saved_models = {}

    for batch_size in BATCH_SIZES:
        print(
            "\n=============================="
        )

        print(
            f"BATCH SIZE {batch_size}"
        )

        print(
            "=============================="
        )

        torch.manual_seed(SEED)

        model = VGGLike().to(DEVICE)

        (
            history,
            best_epoch,
            best_metrics,
            best_weights,
        ) = train_model(
            model,
            data["train"],
            data["validation"],
            batch_size,
        )

        histories[batch_size] = history

        results[batch_size] = {
            "batch_size": batch_size,
            "best_epoch": best_epoch,
            "validation_loss": (
                best_metrics["loss"]
            ),
            "classification_error": (
                best_metrics["error"]
            ),
            "sensitivity": (
                best_metrics["sensitivity"]
            ),
            "specificity": (
                best_metrics["specificity"]
            ),
        }

        saved_models[batch_size] = (
            best_weights
        )

    results_df = pd.DataFrame(
        list(results.values())
    )

    results_df.to_csv(
        OUTPUT_DIR / "validation_results.csv",
        index=False,
    )

    plot_validation_loss(histories)
    plot_validation_accuracy(histories)

    best_row = results_df.loc[
        results_df[
            "classification_error"
        ].idxmin()
    ]

    best_batch_size = int(
        best_row["batch_size"]
    )

    print(
        "\n=============================="
    )

    print("VALIDATION RESULTS")

    print(
        "=============================="
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(
        "\n=============================="
    )

    print("SELECTED MODEL")

    print(
        "=============================="
    )

    print(
        f"Batch size: {best_batch_size}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE:.0e}"
    )

    print(
        f"Best epoch: "
        f"{int(best_row['best_epoch'])}"
    )

    print(
        f"Validation classification error: "
        f"{best_row['classification_error']:.4f}"
    )

    # Load the selected validation checkpoint.
    final_model = VGGLike().to(DEVICE)

    final_model.load_state_dict(
        saved_models[best_batch_size]
    )

    # Evaluate the selected model on the test set once.
    test_metrics = evaluate(
        final_model,
        data["test"],
        best_batch_size,
    )

    torch.save(
        final_model.state_dict(),
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


if __name__ == "__main__":
    main()