"""Sequence-risk model (roadmap B2): a small GRU over an account's recent transactions.

Deliberately small — two layers, 48 hidden units, ~20k parameters. This is a *second
signal*, not a replacement: the calibrated LightGBM point-risk model stays primary
because it is the one that produces SHAP reason codes, and "every block has a reason
code" is a hard requirement in docs/ARCHITECTURE.md §1. A GRU that outscored it would
still not be allowed to overrule it without an explainability story to match.

What it adds is independence from window width. The point-risk model compresses history
into 1h/24h/7d aggregates; slow-ramp evasion works by stretching a burst until those
aggregates go quiet. A model reading the ordered sequence has no window to stretch past.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cerberus.features.sequences import N_SEQUENCE_FEATURES


@dataclass
class SequenceTrainResult:
    model: object
    roc_auc: float
    pr_auc: float
    n_train: int
    n_test: int
    epochs_run: int
    final_train_loss: float


def _require_torch():
    try:
        import torch

        return torch
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "The sequence model needs PyTorch. Install it with:\n"
            '  pip install -e ".[sequence]"\n'
            "or:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "The rest of the pipeline runs without it."
        ) from exc


def build_model(hidden_size: int = 48, num_layers: int = 2, dropout: float = 0.1):
    torch = _require_torch()
    nn = torch.nn

    class SequenceRiskNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_size=N_SEQUENCE_FEATURES,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, hidden = self.gru(x)
            # Last layer's final hidden state summarises the whole window.
            return self.head(hidden[-1]).squeeze(-1)

    return SequenceRiskNet()


def train_sequence_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int = 8,
    batch_size: int = 512,
    learning_rate: float = 3e-3,
    random_seed: int = 1337,
) -> SequenceTrainResult:
    """Train the GRU and report held-out ROC-AUC / PR-AUC.

    Positives are ~2% of rows, so the loss is class-weighted by the empirical imbalance
    — the same reasoning as `class_weight="balanced"` on the LightGBM side. Without it
    the model converges to predicting the majority class and reports a flattering loss
    while being useless.
    """
    torch = _require_torch()
    from sklearn.metrics import average_precision_score, roc_auc_score

    torch.manual_seed(random_seed)
    model = build_model()

    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0), dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(y_train)
    dataset = torch.utils.data.TensorDataset(Xt, yt)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    final_loss = float("nan")
    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        final_loss = epoch_loss / max(n_batches, 1)
        print(f"  epoch {epoch + 1}/{epochs}  loss={final_loss:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test))
        scores = torch.sigmoid(logits).numpy()

    return SequenceTrainResult(
        model=model,
        roc_auc=float(roc_auc_score(y_test, scores)),
        pr_auc=float(average_precision_score(y_test, scores)),
        n_train=len(y_train),
        n_test=len(y_test),
        epochs_run=epochs,
        final_train_loss=final_loss,
    )


def predict_scores(model, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Uncalibrated P(fraud) per row. Calibrate before this reaches the decision layer —
    a raw sigmoid under a class-weighted loss is not a probability."""
    torch = _require_torch()
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size])
            out.append(torch.sigmoid(model(batch)).numpy())
    return np.concatenate(out) if out else np.array([], dtype=np.float32)
