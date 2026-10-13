import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from a3_wide_deep_model import WideDeepPriceModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "hdb_price_prediction.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "wide_deep"

HIDDEN_WIDTH = 128
EMBEDDING_DIM = 12
EPOCHS = 13

LAMBDAS = [0, 0.25, 0.5, 1, 2]
SEEDS = [1, 2, 3, 4, 5]

BASELINE_RMSE_MEAN = 99173.2406
BASELINE_RMSE_STD = 5620.7437

BASELINE_R2_MEAN = 0.6598
BASELINE_R2_STD = 0.0395


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


def prepare_data(csv_path, config, combine_validation=False):
    rows = pd.read_csv(
        csv_path
    ).to_dict(
        orient="records"
    )

    if combine_validation:
        train_years = (
            *config.train_years,
            config.validation_year,
        )
    else:
        train_years = config.train_years

    train_rows = [
        row
        for row in rows
        if int(row["year"]) in train_years
    ]

    validation_rows = [
        row
        for row in rows
        if int(row["year"]) == config.validation_year
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

    result = {
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

    if not combine_validation:
        result["validation"] = make_dataset(
            validation_rows,
            category_maps,
            means,
            stds,
            config,
        )

    return result


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


def find_best_lambda(config):
    data = prepare_data(
        DATA_PATH,
        config,
        combine_validation=False,
    )

    results = []

    print("\n==============================")
    print("LAMBDA SEARCH")
    print("==============================")

    for lambda_value in LAMBDAS:
        torch.manual_seed(config.seed)

        model = WideDeepPriceModel(
            data["cardinalities"],
            data["target_mean"],
            data["target_std"],
            config,
            lambda_value,
        )

        train_model(
            model,
            data["train"],
            config,
        )

        validation_rmse, _ = calculate_metrics(
            model,
            data["validation"],
            config.batch_size,
        )

        results.append(
            {
                "lambda": lambda_value,
                "validation_rmse": validation_rmse,
            }
        )

        print(
            f"lambda {lambda_value:>4} | "
            f"validation RMSE "
            f"{validation_rmse:,.2f}"
        )

    results = pd.DataFrame(results)

    best_row = results.loc[
        results["validation_rmse"].idxmin()
    ]

    best_lambda = float(
        best_row["lambda"]
    )

    results.to_csv(
        OUTPUT_DIR / "lambda_search.csv",
        index=False,
    )

    print(
        f"\nBest lambda: {best_lambda}"
    )

    return best_lambda, results


def evaluate_five_seeds(
    config,
    best_lambda,
):
    data = prepare_data(
        DATA_PATH,
        config,
        combine_validation=True,
    )

    results = []

    print("\n==============================")
    print("FIVE-SEED EVALUATION")
    print("==============================")

    for seed in SEEDS:
        torch.manual_seed(seed)

        model = WideDeepPriceModel(
            data["cardinalities"],
            data["target_mean"],
            data["target_std"],
            config,
            best_lambda,
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


def plot_lambda_search(results):
    plt.figure(figsize=(8, 5))

    plt.plot(
        results["lambda"],
        results["validation_rmse"],
        marker="o",
    )

    plt.xlabel("Lambda")
    plt.ylabel("Validation RMSE (SGD)")
    plt.title("Validation RMSE for Lambda Search")
    plt.grid(True)
    plt.tight_layout()

    path = OUTPUT_DIR / "lambda_search.png"

    plt.savefig(path, dpi=300)
    plt.show()

    print(f"Wrote {path}")


def plot_rmse_comparison(
    wide_mean,
    wide_std,
):
    labels = [
        "A1 Baseline",
        "Wide & Deep",
    ]

    means = [
        BASELINE_RMSE_MEAN,
        wide_mean,
    ]

    errors = [
        BASELINE_RMSE_STD,
        wide_std,
    ]

    plt.figure(figsize=(7, 5))

    plt.bar(
        labels,
        means,
        yerr=errors,
        capsize=5,
    )

    plt.ylabel("Test RMSE (SGD)")
    plt.title("Baseline vs Wide-and-Deep Test RMSE")
    plt.tight_layout()

    path = OUTPUT_DIR / "rmse_comparison.png"

    plt.savefig(path, dpi=300)
    plt.show()

    print(f"Wrote {path}")


def plot_r2_comparison(
    wide_mean,
    wide_std,
):
    labels = [
        "A1 Baseline",
        "Wide & Deep",
    ]

    means = [
        BASELINE_R2_MEAN,
        wide_mean,
    ]

    errors = [
        BASELINE_R2_STD,
        wide_std,
    ]

    plt.figure(figsize=(7, 5))

    plt.bar(
        labels,
        means,
        yerr=errors,
        capsize=5,
    )

    plt.ylabel("Test R²")
    plt.title("Baseline vs Wide-and-Deep Test R²")
    plt.tight_layout()

    path = OUTPUT_DIR / "r2_comparison.png"

    plt.savefig(path, dpi=300)
    plt.show()

    print(f"Wrote {path}")


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

    best_lambda, lambda_results = (
        find_best_lambda(config)
    )

    plot_lambda_search(
        lambda_results
    )

    results = evaluate_five_seeds(
        config,
        best_lambda,
    )

    mean_rmse = results["test_rmse"].mean()
    std_rmse = results["test_rmse"].std(ddof=1)

    mean_r2 = results["test_r2"].mean()
    std_r2 = results["test_r2"].std(ddof=1)

    results.to_csv(
        OUTPUT_DIR / "wide_deep_results.csv",
        index=False,
    )

    print("\n==============================")
    print("FINAL WIDE-AND-DEEP RESULTS")
    print("==============================")

    print(f"Lambda: {best_lambda}")

    print(
        f"Test RMSE: "
        f"{mean_rmse:,.2f} ± "
        f"{std_rmse:,.2f}"
    )

    print(
        f"Test R²: "
        f"{mean_r2:.4f} ± "
        f"{std_r2:.4f}"
    )

    plot_rmse_comparison(
        mean_rmse,
        std_rmse,
    )

    plot_r2_comparison(
        mean_r2,
        std_r2,
    )


if __name__ == "__main__":
    main()