import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from model import PriceModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "hdb_price_prediction.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final"

BEST_HIDDEN_WIDTH = 128
BEST_EMBEDDING_DIM = 12
BEST_EPOCH = 13


def make_dataset(rows, category_maps, means, stds, config):
    categorical = torch.tensor(
        [
            [category_maps[name].get(row[name], 0) for name in config.categorical_features]
            for row in rows
        ],
        dtype=torch.long,
    )

    continuous = torch.tensor(
        [
            [float(row[name]) for name in config.continuous_features]
            for row in rows
        ],
        dtype=torch.float32,
    )
    continuous = (continuous - means) / stds

    target = torch.tensor(
        [float(row["resale_price"]) for row in rows],
        dtype=torch.float32,
    )

    return TensorDataset(categorical, continuous, target)


def prepare_final_data(csv_path, config):
    rows = pd.read_csv(csv_path).to_dict(orient="records")

    train_years = (*config.train_years, config.validation_year)

    train_rows = [
        row for row in rows
        if int(row["year"]) in train_years
    ]
    test_rows = [
        row for row in rows
        if int(row["year"]) == config.test_year
    ]

    category_maps = {}

    for name in config.categorical_features:
        values = sorted({row[name] for row in train_rows})
        category_maps[name] = {
            value: index + 1
            for index, value in enumerate(values)
        }

    train_continuous = torch.tensor(
        [
            [float(row[name]) for name in config.continuous_features]
            for row in train_rows
        ],
        dtype=torch.float32,
    )

    means = train_continuous.mean(dim=0)
    stds = train_continuous.std(dim=0).clamp_min(1e-8)

    train_target = torch.tensor(
        [float(row["resale_price"]) for row in train_rows],
        dtype=torch.float32,
    )

    target_mean = train_target.mean().item()
    target_std = train_target.std(correction=0).clamp_min(1e-8).item()

    cardinalities = [
        len(category_maps[name]) + 1
        for name in config.categorical_features
    ]

    return {
        "train": make_dataset(
            train_rows, category_maps, means, stds, config
        ),
        "test": make_dataset(
            test_rows, category_maps, means, stds, config
        ),
        "cardinalities": cardinalities,
        "target_mean": target_mean,
        "target_std": target_std,
    }


@torch.no_grad()
def calculate_metrics(model, dataset, batch_size):
    model.eval()

    predictions = []
    targets = []

    for categorical, continuous, target in DataLoader(
        dataset,
        batch_size=batch_size,
    ):
        predictions.append(
            model.predict_price(categorical, continuous)
        )
        targets.append(target)

    prediction = torch.cat(predictions)
    target = torch.cat(targets)

    rmse = (
        (prediction - target)
        .square()
        .mean()
        .sqrt()
        .item()
    )

    ss_res = (target - prediction).square().sum()
    ss_tot = (target - target.mean()).square().sum()
    r_squared = (1 - ss_res / ss_tot).item()

    return rmse, r_squared


def train_model(model, train_data, test_data, config):
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
    )

    loss_function = nn.MSELoss()
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    train_history = []
    test_history = []

    for epoch in range(1, BEST_EPOCH + 1):
        model.train()

        for categorical, continuous, target in train_loader:
            optimiser.zero_grad()

            prediction = model(categorical, continuous)
            standard_target = (
                target - model.target_mean
            ) / model.target_std

            loss = loss_function(
                prediction,
                standard_target,
            )

            loss.backward()
            optimiser.step()

        train_rmse, _ = calculate_metrics(
            model,
            train_data,
            config.batch_size,
        )
        test_rmse, _ = calculate_metrics(
            model,
            test_data,
            config.batch_size,
        )

        train_history.append(train_rmse)
        test_history.append(test_rmse)

        print(
            f"epoch {epoch:02d} | "
            f"train RMSE {train_rmse:,.2f} | "
            f"test RMSE {test_rmse:,.2f}"
        )

    return train_history, test_history


def plot_learning_curve(train_history, test_history):
    epochs = range(1, len(train_history) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_history, marker="o", label="Training RMSE")
    plt.plot(epochs, test_history, marker="o", label="Testing RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (SGD)")
    plt.title(
        "Training and Testing RMSE\n"
        f"Hidden Width = {BEST_HIDDEN_WIDTH}, "
        f"Embedding Dimension = {BEST_EMBEDDING_DIM}"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = OUTPUT_DIR / "final_learning_curve.png"
    plt.savefig(path, dpi=300)
    plt.show()

    print(f"Wrote {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = Config()
    config.hidden_width = BEST_HIDDEN_WIDTH
    config.embedding_dims = (BEST_EMBEDDING_DIM,) * 4

    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    data = prepare_final_data(DATA_PATH, config)

    model = PriceModel(
        data["cardinalities"],
        data["target_mean"],
        data["target_std"],
        config,
    )

    train_history, test_history = train_model(
        model,
        data["train"],
        data["test"],
        config,
    )

    test_rmse, test_r_squared = calculate_metrics(
        model,
        data["test"],
        config.batch_size,
    )

    plot_learning_curve(
        train_history,
        test_history,
    )

    torch.save(
        model.state_dict(),
        OUTPUT_DIR / "final_model.pt",
    )

    pd.DataFrame(
        {
            "epoch": range(1, BEST_EPOCH + 1),
            "train_rmse": train_history,
            "test_rmse": test_history,
        }
    ).to_csv(
        OUTPUT_DIR / "final_learning_curve.csv",
        index=False,
    )

    print("\nFINAL RESULTS")
    print(f"Hidden width: {BEST_HIDDEN_WIDTH}")
    print(f"Embedding dimension: {BEST_EMBEDDING_DIM}")
    print(f"Epochs: {BEST_EPOCH}")
    print(f"Test RMSE: {test_rmse:,.2f}")
    print(f"Test R²: {test_r_squared:.4f}")


if __name__ == "__main__":
    main()