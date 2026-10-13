import matplotlib.pyplot as plt

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ray import tune, train
from ray.tune import RunConfig

from config import Config
from data import prepare_data
from model import PriceModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "hdb_price_prediction.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tuning"


@torch.no_grad()
def calculate_rmse(model, dataset, batch_size):
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

    return (
        (prediction - target)
        .square()
        .mean()
        .sqrt()
        .item()
    )


def train_trial(ray_config):
    config = Config()

    # Override the two hyperparameters
    config.hidden_width = ray_config["hidden_width"]

    embedding_dim = ray_config["embedding_dim"]
    config.embedding_dims = (
        embedding_dim,
        embedding_dim,
        embedding_dim,
        embedding_dim,
    )

    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )

    data = prepare_data(DATA_PATH, config)

    model = PriceModel(
        data["cardinalities"],
        data["target_mean"],
        data["target_std"],
        config,
    )

    train_loader = DataLoader(
        data["train"],
        batch_size=config.batch_size,
        shuffle=True,
    )

    loss_function = nn.MSELoss()

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    best_validation_rmse = float("inf")

    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()

        squared_error = 0.0
        row_count = 0

        for categorical, continuous, target in train_loader:
            optimiser.zero_grad()

            standard_prediction = model(
                categorical,
                continuous,
            )

            standard_target = (
                target - model.target_mean
            ) / model.target_std

            loss = loss_function(
                standard_prediction,
                standard_target,
            )

            loss.backward()
            optimiser.step()

            prediction = (
                standard_prediction.detach()
                * model.target_std
                + model.target_mean
            )

            squared_error += (
                (prediction - target)
                .square()
                .sum()
                .item()
            )

            row_count += len(target)

        train_rmse = (
            squared_error / row_count
        ) ** 0.5

        validation_rmse = calculate_rmse(
            model,
            data["validation"],
            config.batch_size,
        )

        # ----------------------------------------
        # Track best validation RMSE
        # ----------------------------------------
        improved = (
            validation_rmse
            < best_validation_rmse
            - config.early_stopping_threshold
        )

        if improved:
            best_validation_rmse = validation_rmse
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # ----------------------------------------
        # Send epoch's results to Ray
        # ----------------------------------------
        tune.report(
            {
                "epoch": epoch,
                "train_rmse": train_rmse,
                "validation_rmse": validation_rmse,
                "best_validation_rmse": best_validation_rmse,
            }
        )

        if (
            epochs_without_improvement
            == config.early_stopping_patience
        ):
            break


def plot_best_learning_curve(best_result):
    history = best_result.metrics_dataframe

    epochs = history["epoch"]
    train_rmse = history["train_rmse"]
    validation_rmse = history["validation_rmse"]

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_rmse,
        marker="o",
        label="Training RMSE",
    )

    plt.plot(
        epochs,
        validation_rmse,
        marker="o",
        label="Validation RMSE",
    )

    plt.xlabel("Epoch")
    plt.ylabel("RMSE (SGD)")
    plt.title(
        "Training and Validation RMSE\n"
        f"Hidden Width = {best_result.config['hidden_width']}, "
        f"Embedding Dimension = {best_result.config['embedding_dim']}"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "best_learning_curve.png"
    plt.savefig(output_path, dpi=300)

    print(f"Wrote {output_path}")

    plt.show()


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    search_space = {
        "hidden_width": tune.grid_search(
            [16, 32, 64, 128, 256]
        ),
        "embedding_dim": tune.grid_search(
            [2, 4, 8, 12, 16]
        ),
    }

    tuner = tune.Tuner(
        train_trial,
        param_space=search_space,
        run_config=RunConfig(
            name="price_model_grid_search",
            storage_path=str(OUTPUT_DIR.resolve()),
            verbose=1,
        ),
    )

    results = tuner.fit()

    # ----------------------------------------
    # Find configuration with lowest
    #    best validation RMSE
    # ----------------------------------------
    best_result = results.get_best_result(
        metric="best_validation_rmse",
        mode="min",
        scope="all",
    )

    plot_best_learning_curve(best_result)

    print("\n==============================")
    print("BEST CONFIGURATION")
    print("==============================")

    print(
        f"Hidden width: "
        f"{best_result.config['hidden_width']}"
    )

    print(
        f"Embedding dimension: "
        f"{best_result.config['embedding_dim']}"
    )

    print(
        f"Best validation RMSE: "
        f"{best_result.metrics['best_validation_rmse']:,.2f}"
    )

    # ----------------------------------------
    # 9. Print all 25 configurations
    # ----------------------------------------
    print("\n==============================")
    print("ALL RESULTS")
    print("==============================")

    rows = []

    for result in results:
        if result.error:
            continue
        best_rmse = result.metrics_dataframe[
            "best_validation_rmse"
        ].min()
        rows.append(
            (
                result.config["hidden_width"],
                result.config["embedding_dim"],
                best_rmse,
            )
        )

    # Sort by validation RMSE
    rows.sort(key=lambda x: x[2])

    print(
        f"{'Hidden':>10} "
        f"{'Embedding':>12} "
        f"{'Best Val RMSE':>18}"
    )

    for hidden_width, embedding_dim, rmse in rows:
        print(
            f"{hidden_width:>10} "
            f"{embedding_dim:>12} "
            f"{rmse:>18,.2f}"
        )


if __name__ == "__main__":
    main()