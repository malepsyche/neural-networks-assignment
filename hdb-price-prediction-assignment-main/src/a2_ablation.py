import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from model import PriceModel
from a2_ablation_models import (
    ContinuousOnlyPriceModel,
    SigmoidPriceModel,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "hdb_price_prediction.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ablation"

BEST_HIDDEN_WIDTH = 128
BEST_EMBEDDING_DIM = 12
BEST_EPOCH = 13

SEEDS = [1, 2, 3, 4, 5]


def make_dataset(
    rows,
    category_maps,
    means,
    stds,
    config,
):
    categorical = torch.tensor(
        [
            [
                category_maps[name].get(row[name], 0)
                for name in config.categorical_features
            ]
            for row in rows
        ],
        dtype=torch.long,
    )

    continuous = torch.tensor(
        [
            [
                float(row[name])
                for name in config.continuous_features
            ]
            for row in rows
        ],
        dtype=torch.float32,
    )

    continuous = (continuous - means) / stds

    target = torch.tensor(
        [float(row["resale_price"]) for row in rows],
        dtype=torch.float32,
    )

    return TensorDataset(
        categorical,
        continuous,
        target,
    )


def prepare_final_data(csv_path, config):
    rows = pd.read_csv(
        csv_path
    ).to_dict(
        orient="records"
    )

    train_years = (
        *config.train_years,
        config.validation_year,
    )

    train_rows = [
        row
        for row in rows
        if int(row["year"]) in train_years
    ]

    test_rows = [
        row
        for row in rows
        if int(row["year"]) == config.test_year
    ]

    category_maps = {}

    for name in config.categorical_features:
        values = sorted(
            {row[name] for row in train_rows}
        )

        category_maps[name] = {
            value: index + 1
            for index, value in enumerate(values)
        }

    train_continuous = torch.tensor(
        [
            [
                float(row[name])
                for name in config.continuous_features
            ]
            for row in train_rows
        ],
        dtype=torch.float32,
    )

    means = train_continuous.mean(dim=0)

    stds = (
        train_continuous
        .std(dim=0)
        .clamp_min(1e-8)
    )

    train_target = torch.tensor(
        [
            float(row["resale_price"])
            for row in train_rows
        ],
        dtype=torch.float32,
    )

    target_mean = (
        train_target
        .mean()
        .item()
    )

    target_std = (
        train_target
        .std(correction=0)
        .clamp_min(1e-8)
        .item()
    )

    cardinalities = [
        len(category_maps[name]) + 1
        for name in config.categorical_features
    ]

    return {
        "train": make_dataset(
            train_rows,
            category_maps,
            means,
            stds,
            config,
        ),
        "test": make_dataset(
            test_rows,
            category_maps,
            means,
            stds,
            config,
        ),
        "cardinalities": cardinalities,
        "target_mean": target_mean,
        "target_std": target_std,
    }


@torch.no_grad()
def calculate_metrics(
    model,
    dataset,
    batch_size,
):
    model.eval()

    predictions = []
    targets = []

    for categorical, continuous, target in DataLoader(
        dataset,
        batch_size=batch_size,
    ):
        predictions.append(
            model.predict_price(
                categorical,
                continuous,
            )
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

    ss_res = (
        (target - prediction)
        .square()
        .sum()
    )

    ss_tot = (
        (target - target.mean())
        .square()
        .sum()
    )

    r_squared = (
        1 - ss_res / ss_tot
    ).item()

    return rmse, r_squared


def create_model(
    model_name,
    data,
    config,
):
    if model_name == "Baseline":
        return PriceModel(
            data["cardinalities"],
            data["target_mean"],
            data["target_std"],
            config,
        )

    if model_name == "No Embeddings":
        return ContinuousOnlyPriceModel(
            data["target_mean"],
            data["target_std"],
            config,
        )

    if model_name == "Sigmoid":
        return SigmoidPriceModel(
            data["cardinalities"],
            data["target_mean"],
            data["target_std"],
            config,
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def train_model(
    model,
    train_data,
    config,
):
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

    for _ in range(BEST_EPOCH):
        model.train()

        for categorical, continuous, target in train_loader:
            optimiser.zero_grad()

            prediction = model(
                categorical,
                continuous,
            )

            standard_target = (
                target - model.target_mean
            ) / model.target_std

            loss = loss_function(
                prediction,
                standard_target,
            )

            loss.backward()
            optimiser.step()


def run_experiments(
    data,
    config,
):
    results = []

    model_names = [
        "Baseline",
        "No Embeddings",
        "Sigmoid",
    ]

    for model_name in model_names:
        print(
            f"\n==============================\n"
            f"{model_name}\n"
            f"=============================="
        )

        for seed in SEEDS:
            torch.manual_seed(seed)

            model = create_model(
                model_name,
                data,
                config,
            )

            train_model(
                model,
                data["train"],
                config,
            )

            rmse, r_squared = calculate_metrics(
                model,
                data["test"],
                config.batch_size,
            )

            results.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "test_rmse": rmse,
                    "test_r2": r_squared,
                }
            )

            print(
                f"seed {seed} | "
                f"RMSE {rmse:,.2f} | "
                f"R² {r_squared:.4f}"
            )

    return pd.DataFrame(results)


def summarise_results(results):
    summary = (
        results
        .groupby("model")
        .agg(
            mean_rmse=("test_rmse", "mean"),
            std_rmse=("test_rmse", "std"),
            mean_r2=("test_r2", "mean"),
            std_r2=("test_r2", "std"),
        )
        .reset_index()
    )

    return summary


def plot_rmse(summary):
    plt.figure(figsize=(8, 5))

    plt.bar(
        summary["model"],
        summary["mean_rmse"],
        yerr=summary["std_rmse"],
        capsize=5,
    )

    plt.ylabel("Test RMSE (SGD)")
    plt.title("Test RMSE Across Five Seeds")
    plt.tight_layout()

    path = OUTPUT_DIR / "ablation_rmse.png"

    plt.savefig(
        path,
        dpi=300,
    )

    plt.show()

    print(f"Wrote {path}")


def plot_r2(summary):
    plt.figure(figsize=(8, 5))

    plt.bar(
        summary["model"],
        summary["mean_r2"],
        yerr=summary["std_r2"],
        capsize=5,
    )

    plt.ylabel("Test R²")
    plt.title("Test R² Across Five Seeds")
    plt.tight_layout()

    path = OUTPUT_DIR / "ablation_r2.png"

    plt.savefig(
        path,
        dpi=300,
    )

    plt.show()

    print(f"Wrote {path}")


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = Config()

    config.hidden_width = BEST_HIDDEN_WIDTH
    config.embedding_dims = (
        BEST_EMBEDDING_DIM,
    ) * 4

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )

    data = prepare_final_data(
        DATA_PATH,
        config,
    )

    results = run_experiments(
        data,
        config,
    )

    summary = summarise_results(
        results
    )

    results.to_csv(
        OUTPUT_DIR / "ablation_results.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "ablation_summary.csv",
        index=False,
    )

    print(
        "\n=============================="
    )
    print(
        "ABLATION SUMMARY"
    )
    print(
        "=============================="
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:,.4f}",
        )
    )

    plot_rmse(summary)
    plot_r2(summary)


if __name__ == "__main__":
    main()