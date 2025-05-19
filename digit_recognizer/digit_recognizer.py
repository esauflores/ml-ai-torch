# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair==5.5.0",
#     "marimo",
#     "numpy==2.2.5",
#     "openai==1.78.1",
#     "pandas==2.2.3",
#     "pillow==11.2.1",
#     "plotly==6.0.1",
#     "polars==1.29.0",
#     "pympler==1.1",
#     "pyobsplot==0.5.3.2",
#     "scikit-learn==1.6.1",
#     "torch==2.7.0",
#     "torchvision==0.22.0",
#     "tqdm==4.67.1",
# ]
# ///

import marimo

__generated_with = "0.13.9"
app = marimo.App(
    width="columns",
    layout_file="layouts/digit_recognizer.grid.json",
)


@app.cell(column=0)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""# MNIST Digit Recognizer""").center(),
            # mo.md(r"""## Author: Cesar Flores""").center(),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""# Data loading""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Train data""")
    return


@app.cell
def _(alt, chart_bg_color, mo, train_df_with_img):
    _chart = (
        alt.Chart(train_df_with_img)
        .mark_bar()
        .encode(
            x=alt.X(
                "label:O",
                title="Digits",
                axis=alt.Axis(labelAngle=0, titlePadding=15),
            ),
            y=alt.Y(
                "count()",
                title="Frequency",
                axis=alt.Axis(grid=False, titlePadding=15),
            ),
            color=alt.Color("label:N", scale=alt.Scale(scheme="category10")),
            tooltip=[
                alt.Tooltip("label:N", title="Digit"),
                alt.Tooltip("count()", title="Count"),
            ],
        )
        .properties(title="Digit Frequency Distribution in Train Data")
        .configure(background=chart_bg_color)
        .configure_legend(orient="top", title=None, offset=20)
        .configure_title(fontSize=16, offset=20)
    )

    digit_train_data_chart = mo.ui.altair_chart(
        _chart,
        chart_selection="point",
        legend_selection=False,
    )

    digit_train_data_chart
    return (digit_train_data_chart,)


@app.cell
def _(base64_to_image, digit_train_data_chart, mo):
    _selected_point = digit_train_data_chart.selections.get("select_point")

    _data = digit_train_data_chart.value
    if not _selected_point:
        _data = digit_train_data_chart.dataframe  # all points

    _data = _data.reset_index()
    _data = _data.rename(columns={"label": "Digit", "base64_img": "Image", "index": "Index"})
    _data = _data.to_dict(orient="records")

    mo.ui.table(
        _data,
        format_mapping={"Image": base64_to_image},
        selection=None,
        show_download=False,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# Evaluation Phase""")
    return


@app.cell
def _(alt, chart_bg_color, eval_metrics, mo, pd):
    _train_loss = eval_metrics["train_loss"]
    _test_loss = eval_metrics["test_loss"]
    _epochs = list(range(1, len(_train_loss) + 1))

    df_loss = pd.DataFrame(
        {
            "Epoch": _epochs * 2,
            "Loss": _train_loss + _test_loss,
            "Dataset": ["Train"] * len(_train_loss) + ["Test"] * len(_test_loss),
        }
    )

    loss_chart = (
        alt.Chart(df_loss)
        .mark_line(point=True)
        .encode(x="Epoch", y="Loss", color="Dataset")
        .properties(title="Loss over epochs")
        .configure(background=chart_bg_color)
        .configure_legend(orient="top", title=None, offset=20)
        .configure_title(fontSize=16, offset=20)
    )

    mo.ui.altair_chart(loss_chart)
    return


@app.cell
def _(alt, chart_bg_color, eval_metrics, mo, pd):
    _train_acc = eval_metrics["train_acc"]
    _test_acc = eval_metrics["test_acc"]
    _epochs = list(range(1, len(eval_metrics["test_acc"]) + 1))

    df_acc = pd.DataFrame(
        {
            "Epoch": _epochs * 2,
            "Accuracy": _train_acc + _test_acc,
            "Dataset": ["Train"] * len(_train_acc) + ["Test"] * len(_test_acc),
        }
    )

    _acc_chart = (
        alt.Chart(df_acc)
        .mark_line(point=True)
        .encode(x="Epoch", y="Accuracy", color="Dataset")
        .properties(title="Accuracy over epochs")
        .configure(background=chart_bg_color)
        .configure_legend(orient="top", title=None, offset=20)
        .configure_title(fontSize=16, offset=20)
    )

    acc_chart = mo.ui.altair_chart(_acc_chart, chart_selection="interval")

    acc_chart
    return


@app.cell
def _(eval_phase_done, mo):
    eval_phase_done
    mo.md(r"""# Training Phase""")
    return


@app.cell
def _(base64_to_image, mo, test_df_with_img_pred):
    _data = test_df_with_img_pred.copy()

    _data = _data.rename(
        columns={
            "index": "Index",
            "base64_img": "Image",
            "predicted": "Predicted Digit",
        }
    )

    _data = _data.to_dict(orient="records")

    mo.ui.table(
        _data,
        format_mapping={"Image": base64_to_image},
        selection=None,
        show_download=False,
    )
    return


@app.cell
def _(mo):
    mo.md("""Thank you""")
    return


@app.cell(column=1)
def _():
    import os

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return (os,)


@app.cell
def _():
    # Standard library imports
    import base64
    import copy
    import io
    import random
    from typing import Callable, Dict, List, Optional, Tuple

    # Third-party imports
    import altair as alt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, Subset, TensorDataset
    from tqdm.auto import tqdm  # auto choose the right tqdm version
    from sklearn.model_selection import train_test_split
    import marimo as mo
    return (
        Callable,
        DataLoader,
        Dict,
        F,
        GradScaler,
        List,
        Optional,
        TensorDataset,
        Tuple,
        alt,
        autocast,
        base64,
        copy,
        io,
        mo,
        nn,
        np,
        optim,
        pd,
        random,
        torch,
        train_test_split,
    )


@app.cell
def _(os):
    PROJECT_PATH = "~/Python/test/digit_recognizer"
    os.chdir(os.path.expanduser(PROJECT_PATH))  # move to the project directory
    return


@app.cell
def _(device, np, random, torch):
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        torch.use_deterministic_algorithms(True)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # for multi-GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        generator = torch.Generator(device=device).manual_seed(seed)

        return generator


    seed = 42
    generator = set_seed(seed)
    return generator, seed


@app.cell
def _(torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    return (device,)


@app.cell
def _(mo):
    chart_bg_color = "#181C1A" if mo.app_meta().theme == "dark" else "#ffffff"
    return (chart_bg_color,)


@app.cell
def _(Image, base64, io, np):
    def pixels_to_base64(pixels, shape=(28, 28)):
        """
        Convert flattened pixels to a base64-encoded image string.

        Parameters:
        - pixels: 1D array-like of pixel values (must match shape)
        - shape: tuple (height, width) for reshaping the image

        Returns:
        - base64 encoded string of the image
        """

        # Reshape to 2D image with the given shape
        image_array = np.array(pixels).reshape(shape).astype(np.uint8)

        # Create a grayscale image
        image = Image.fromarray(image_array, mode="L")

        # Save image to a BytesIO buffer
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")

        # Encode the image to base64
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img_base64
    return (pixels_to_base64,)


@app.cell
def _(mo):
    def base64_to_image(base64_str, width=28, height=28):
        return mo.image(
            f"data:image/png;base64,{base64_str}",
            width=width,
            height=height,
        )
    return (base64_to_image,)


@app.cell
def _(nn, torch):
    class MNISTNet(nn.Module):
        def __init__(self, n_hidden_neurons):
            super().__init__()
            self.fc1 = torch.nn.Linear(28 * 28, n_hidden_neurons)
            self.act1 = torch.nn.LeakyReLU()
            self.fc2 = torch.nn.Linear(n_hidden_neurons, 10)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act1(x)
            return self.fc2(x)
    return (MNISTNet,)


@app.cell
def _(F, nn, torch):
    def predict(model: nn.Module, X_batch: torch.Tensor) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            logits = model(X_batch)
            probs = F.softmax(logits, dim=1)
        return probs
    return (predict,)


@app.cell
def _(nn, predict, torch):
    def predict_labels(model: nn.Module, X_batch: torch.Tensor) -> torch.Tensor:
        probs = predict(model, X_batch)
        preds = torch.argmax(probs, dim=1)
        return preds
    return (predict_labels,)


@app.cell
def _(GradScaler, autocast, nn, optim, torch):
    def train_batch(
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        device: torch.device,
        scaler: GradScaler,
    ) -> float:
        model.train()
        optimizer.zero_grad()

        with autocast(device_type=device.type):
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return loss.item()
    return (train_batch,)


@app.cell
def _(DataLoader, nn, torch):
    def evaluate_metrics(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> tuple[float, float]:
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = (
                    X_batch.to(device, non_blocking=True),
                    y_batch.to(device, non_blocking=True),
                )
                X_batch = X_batch.view(X_batch.size(0), -1)  # Flatten if needed

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    return (evaluate_metrics,)


@app.cell
def _(
    Callable,
    DataLoader,
    Dict,
    GradScaler,
    List,
    Optional,
    Tuple,
    copy,
    evaluate_metrics,
    mo,
    nn,
    optim,
    torch,
    train_batch,
):
    def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader],
        criterion: Callable,
        optimizer: optim.Optimizer,
        scaler: GradScaler,
        device: torch.device,
        mode: str = "eval",
        n_epochs: int = 20,
        patience: int = 5,
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        if mode not in ["train", "eval"]:
            raise ValueError("mode should be 'train' or 'eval'")

        if mode == "train" and test_loader is not None:
            raise ValueError("test_loader should be None when mode is 'train'")

        if mode == "eval" and test_loader is None:
            raise ValueError("test_loader should be provided when mode is 'eval'")

        if mode == "train":
            print("Training mode: training without validation")
        elif mode == "eval":
            print("Evaluation mode: training with validation")

        metrics: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
        }

        best_test_loss: float = float("inf")
        patience_counter: int = 0
        best_model_state = None

        # pbar = tqdm(range(n_epochs), desc="Training", ncols=120)  # 200 cols

        with mo.status.progress_bar(range(n_epochs), title="Training epoch 1") as bar:
            for epoch in range(n_epochs):
                model.train()
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = (
                        X_batch.to(device, non_blocking=True),
                        y_batch.to(device, non_blocking=True),
                    )

                    train_batch(
                        model,
                        optimizer,
                        criterion,
                        X_batch,
                        y_batch,
                        device,
                        scaler,
                    )

                if mode == "train":
                    train_loss, _ = evaluate_metrics(model, train_loader, criterion, device)
                    metrics["train_loss"].append(train_loss)

                    bar.update(title=f"Training epoch {epoch + 1}", subtitle=f"loss: {train_loss:.4f}")

                elif mode == "eval":
                    train_loss, train_acc = evaluate_metrics(model, train_loader, criterion, device)
                    test_loss, test_acc = evaluate_metrics(model, test_loader, criterion, device)

                    metrics["train_loss"].append(train_loss)
                    metrics["train_acc"].append(train_acc)
                    metrics["test_loss"].append(test_loss)
                    metrics["test_acc"].append(test_acc)

                    bar.update(
                        title=f"Training epoch {epoch + 1}",
                        subtitle=(
                            f"Train loss: {train_loss:.4f} | "
                            f"Train acc: {train_acc:.4f} | "
                            f"Test loss: {test_loss:.4f} | "
                            f"Test acc: {test_acc:.4f}"
                        ),
                    )

                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        best_model_state = copy.deepcopy(model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"⏹️ Early stopping triggered at epoch {epoch + 1}")
                            break

        if best_model_state:
            model.load_state_dict(best_model_state)

        return model, metrics
    return (train_model,)


@app.cell(column=2)
def _(pd):
    # # Convert to parquet --> faster reads
    # train_df = pd.read_csv("train.csv")
    # train_df.to_parquet("train.parquet")

    train_df = pd.read_parquet("train.parquet")
    train_df
    return (train_df,)


@app.cell
def _(pd):
    # Convert to parquet --> faster reads
    # test_df = pd.read_csv("test.csv")
    # test_df.to_parquet("test.parquet")

    test_df = pd.read_parquet("test.parquet")
    test_df
    return (test_df,)


@app.cell
def _(os, pd, pixels_to_base64, train_df):
    _parquet_path = "train_df_with_img.parquet"
    _rebuild = False  # Set to True if you want to force rebuilding

    if not os.path.exists(_parquet_path):
        _rebuild = True

    if _rebuild:
        train_df_with_img = pd.DataFrame(
            {
                "label": train_df["label"],
                "base64_img": train_df.drop(columns="label").apply(pixels_to_base64, axis=1),
            }
        )

        train_df_with_img.to_parquet(_parquet_path)
    else:
        train_df_with_img = pd.read_parquet(_parquet_path)
    return (train_df_with_img,)


@app.cell
def _(os, pd, pixels_to_base64, test_df):
    _parquet_path = "test_df_with_img.parquet"
    _rebuild = False  # Set to True if you want to force rebuilding

    if not os.path.exists(_parquet_path):
        _rebuild = True

    if _rebuild:
        test_df_with_img = pd.DataFrame(
            {
                "base64_img": test_df.apply(pixels_to_base64, axis=1),
            }
        )

        test_df_with_img.to_parquet(_parquet_path)
    else:
        test_df_with_img = pd.read_parquet(_parquet_path)
    return (test_df_with_img,)


@app.cell
def _(
    DataLoader,
    TensorDataset,
    device,
    generator,
    seed,
    torch,
    train_df,
    train_test_split,
):
    _X = train_df.drop("label", axis=1).to_numpy(dtype="float32")
    _y = train_df["label"].to_numpy(dtype="int64")
    _test_size = 0.2

    _X_train, _X_test, _y_train, _y_test = train_test_split(_X, _y, test_size=_test_size, random_state=seed, shuffle=True)

    _X_train_tensor = torch.tensor(_X_train).to(device)
    _y_train_tensor = torch.tensor(_y_train).to(device)
    _X_test_tensor = torch.tensor(_X_test).to(device)
    _y_test_tensor = torch.tensor(_y_test).to(device)

    train_dataset = TensorDataset(_X_train_tensor, _y_train_tensor)
    test_dataset = TensorDataset(_X_test_tensor, _y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, generator=generator)
    return test_loader, train_loader


@app.cell
def _(
    GradScaler,
    MNISTNet,
    device,
    nn,
    optim,
    test_loader,
    train_loader,
    train_model,
):
    # create the model
    _n_hidden_neurons = 128
    eval_model = MNISTNet(_n_hidden_neurons)
    _criterion = nn.CrossEntropyLoss()
    _optimizer = optim.AdamW(eval_model.parameters(), lr=1e-3, weight_decay=1e-4)
    _scaler = GradScaler()

    eval_model, eval_metrics = train_model(
        model=eval_model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=_criterion,
        optimizer=_optimizer,
        scaler=_scaler,
        n_epochs=20,
        patience=5,
        device=device,
        mode="eval",
    )

    eval_phase_done = True
    return eval_metrics, eval_phase_done


@app.cell
def _(
    DataLoader,
    TensorDataset,
    device,
    eval_phase_done,
    generator,
    torch,
    train_df,
):
    eval_phase_done  # needs to do evaluation phase before

    _X = train_df.drop("label", axis=1).to_numpy(dtype="float32")
    _y = train_df["label"].to_numpy(dtype="int64")

    _X_train_tensor = torch.tensor(_X).to(device)
    _y_train_tensor = torch.tensor(_y).to(device)

    all_train_dataset = TensorDataset(_X_train_tensor, _y_train_tensor)
    all_train_loader = DataLoader(all_train_dataset, batch_size=512, shuffle=True, generator=generator)
    return (all_train_loader,)


@app.cell
def _(GradScaler, MNISTNet, all_train_loader, device, nn, optim, train_model):
    _n_hidden_neurons = 128
    final_model = MNISTNet(_n_hidden_neurons)

    _criterion = nn.CrossEntropyLoss()
    _optimizer = optim.AdamW(final_model.parameters(), lr=1e-3, weight_decay=1e-4)
    _scaler = GradScaler()

    final_model, _ = train_model(
        model=final_model,
        train_loader=all_train_loader,
        test_loader=None,
        criterion=_criterion,
        optimizer=_optimizer,
        scaler=_scaler,
        n_epochs=30,
        device=device,
        mode="train",
    )
    return (final_model,)


@app.cell
def _(
    DataLoader,
    device,
    final_model,
    generator,
    pd,
    predict_labels,
    test_df,
    test_df_with_img,
    torch,
):
    _X_test = test_df.to_numpy(dtype="float32")
    _X_test_tensor = torch.tensor(_X_test).to(device)
    final_test_loader = DataLoader(_X_test_tensor, batch_size=512, shuffle=False, generator=generator)

    predictions = []
    for _X_batch in final_test_loader:
        _X_batch = _X_batch.view(_X_batch.size(0), -1)  # Flatten if needed
        preds = predict_labels(final_model, _X_batch)
        predictions.extend(preds.cpu().numpy())

    test_df_with_img_pred = pd.DataFrame(
        {
            "index": test_df_with_img.index,  # add index as a column
            "base64_img": test_df_with_img["base64_img"],
            "predicted": predictions,
        }
    )
    return (test_df_with_img_pred,)


@app.cell
def _(mo):
    n = int(1e6)
    with mo.status.progress_bar(range(n), title="Training epoch 1") as bar:
        for _ in range(n):
            bar.update()
    return


@app.cell
def _():
    n2 = int(1e8)
    a = 0
    for _ in range(n2):
        a+= 1

    a
    return


if __name__ == "__main__":
    app.run()
