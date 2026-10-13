import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from captum.attr import FeatureAblation

from config import Config
from model import PriceModel
from a3_wide_deep_model import WideDeepPriceModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "hdb_price_prediction.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "feature_ablation"

HIDDEN_WIDTH = 128
EMBEDDING_DIM = 12
EPOCHS = 13
LAMBDA = 2.0
SEED = 42


def make_dataset(rows, category_maps, means, stds, config):
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
    stds = train_continuous.std(dim=0).clamp_min(1e-8)

    train_target = torch.tensor(
        [float(row["resale_price"]) for row in train_rows],
        dtype=torch.float32,
    )

    target_mean = train_target.mean().item()

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


def train_model(model, train_data, config):
    loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
    )

    loss_function = nn.MSELoss()

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    for _ in range(EPOCHS):
        model.train()

        for categorical, continuous, target in loader:
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


class PriceOutputWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        categorical,
        continuous,
    ):
        return self.model.predict_price(
            categorical,
            continuous,
        )


def calculate_feature_importance(
    model,
    test_data,
    config,
):
    model.eval()

    wrapper = PriceOutputWrapper(model)
    ablation = FeatureAblation(wrapper)

    loader = DataLoader(
        test_data,
        batch_size=config.batch_size,
        shuffle=False,
    )

    categorical_sum = torch.zeros(
        len(config.categorical_features)
    )

    continuous_sum = torch.zeros(
        len(config.continuous_features)
    )

    sample_count = 0

    for categorical, continuous, _ in loader:
        categorical_baseline = torch.zeros_like(
            categorical
        )

        continuous_baseline = torch.zeros_like(
            continuous
        )

        categorical_mask = torch.arange(
            len(config.categorical_features)
        ).unsqueeze(0)

        continuous_mask = (
            torch.arange(
                len(config.continuous_features)
            )
            .unsqueeze(0)
            + len(config.categorical_features)
        )

        categorical_attr, continuous_attr = (
            ablation.attribute(
                inputs=(
                    categorical,
                    continuous,
                ),
                baselines=(
                    categorical_baseline,
                    continuous_baseline,
                ),
                feature_mask=(
                    categorical_mask,
                    continuous_mask,
                ),
            )
        )

        categorical_sum += (
            categorical_attr
            .abs()
            .sum(dim=0)
        )

        continuous_sum += (
            continuous_attr
            .abs()
            .sum(dim=0)
        )

        sample_count += len(categorical)

    categorical_importance = (
        categorical_sum / sample_count
    )

    continuous_importance = (
        continuous_sum / sample_count
    )

    feature_names = (
        list(config.categorical_features)
        + list(config.continuous_features)
    )

    importance = torch.cat(
        (
            categorical_importance,
            continuous_importance,
        )
    )

    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance.numpy(),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )


def plot_importance(
    results,
    model_name,
    filename,
):
    plt.figure(figsize=(10, 6))

    plt.bar(
        results["feature"],
        results["importance"],
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.xlabel("Feature")
    plt.ylabel(
        "Mean Absolute Attribution (SGD)"
    )

    plt.title(
        f"{model_name} Feature Importance"
    )

    plt.tight_layout()

    path = OUTPUT_DIR / filename

    plt.savefig(
        path,
        dpi=300,
    )

    plt.show()

    print(f"Wrote {path}")


def print_results(
    model_name,
    results,
):
    print(
        f"\n=============================="
    )

    print(model_name)

    print(
        "=============================="
    )

    for _, row in results.iterrows():
        print(
            f"{row['feature']:<25} "
            f"${row['importance']:>12,.2f}"
        )

    print("\nTop 3:")

    for index, (_, row) in enumerate(
        results.head(3).iterrows(),
        start=1,
    ):
        print(
            f"{index}. "
            f"{row['feature']} "
            f"(${row['importance']:,.2f})"
        )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = Config()

    config.hidden_width = HIDDEN_WIDTH

    config.embedding_dims = (
        EMBEDDING_DIM,
    ) * 4

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )

    data = prepare_final_data(
        DATA_PATH,
        config,
    )

    torch.manual_seed(SEED)

    baseline = PriceModel(
        data["cardinalities"],
        data["target_mean"],
        data["target_std"],
        config,
    )

    train_model(
        baseline,
        data["train"],
        config,
    )

    torch.manual_seed(SEED)

    wide_deep = WideDeepPriceModel(
        data["cardinalities"],
        data["target_mean"],
        data["target_std"],
        config,
        LAMBDA,
    )

    train_model(
        wide_deep,
        data["train"],
        config,
    )

    baseline_results = calculate_feature_importance(
        baseline,
        data["test"],
        config,
    )

    wide_deep_results = calculate_feature_importance(
        wide_deep,
        data["test"],
        config,
    )

    baseline_results.to_csv(
        OUTPUT_DIR / "baseline_importance.csv",
        index=False,
    )

    wide_deep_results.to_csv(
        OUTPUT_DIR / "wide_deep_importance.csv",
        index=False,
    )

    print_results(
        "A1 Baseline",
        baseline_results,
    )

    print_results(
        "A3 Wide-and-Deep",
        wide_deep_results,
    )

    plot_importance(
        baseline_results,
        "A1 Baseline",
        "baseline_feature_importance.png",
    )

    plot_importance(
        wide_deep_results,
        "A3 Wide-and-Deep",
        "wide_deep_feature_importance.png",
    )


if __name__ == "__main__":
    main()