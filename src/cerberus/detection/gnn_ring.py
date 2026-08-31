"""GNN ring detector (roadmap B1): GraphSAGE over the entity-link graph.

Runs **alongside** Louvain, never instead of it. Louvain stays the default because it is
explainable to an analyst in one sentence ("these accounts form a dense cluster of shared
devices and cards") and was validated against ground truth. The GNN answers a different
question — the panel one: *"Louvain on synthetic data is easy, would a learned detector
do better?"* — and answers it with a number rather than a claim.

The design pressure here is the mirror of the point-risk regression documented in
docs/EXPERIMENT_ADVANCED_TRAINING.md: graph features made the *point-risk* model brittle
to identity rotation, because a per-transaction classifier shouldn't stake its signal on
structure an attacker can dissolve. A graph model is the right consumer for those same
features — it is supposed to reason about structure, and when structure is destroyed it
should say so rather than quietly mispredicting.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

# Node features, in order. Kept to five cheap, interpretable quantities: a GNN's value
# here is supposed to come from message passing over the graph, not from a rich feature
# vector that would let it succeed while ignoring the topology entirely.
NODE_FEATURE_NAMES = [
    "degree",
    "weighted_degree",
    "component_size",
    "clustering_coefficient",
    "mean_neighbour_degree",
]


@dataclass
class GNNTrainResult:
    model: object
    node_index: dict[str, int]
    roc_auc: float
    pr_auc: float
    n_train_nodes: int
    n_test_nodes: int
    epochs_run: int
    final_loss: float


def _require_torch_geometric():
    try:
        import torch
        import torch_geometric  # noqa: F401

        return torch
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "The GNN ring detector needs PyTorch and PyTorch Geometric:\n"
            '  pip install -e ".[gnn]"\n'
            "The rest of the pipeline, including the Louvain ring detector, runs "
            "without them."
        ) from exc


def build_node_features(graph: nx.Graph) -> tuple[np.ndarray, dict[str, int]]:
    """Feature matrix plus the node -> row-index mapping used everywhere downstream."""
    nodes = sorted(graph.nodes())
    node_index = {node: i for i, node in enumerate(nodes)}

    degrees = dict(graph.degree())
    weighted = dict(graph.degree(weight="weight"))
    clustering = nx.clustering(graph)
    component_of = {n: len(c) for c in nx.connected_components(graph) for n in c}

    rows = []
    for node in nodes:
        neighbours = list(graph.neighbors(node))
        mean_neighbour_degree = float(np.mean([degrees[n] for n in neighbours])) if neighbours else 0.0
        rows.append(
            [
                float(degrees[node]),
                float(weighted[node]),
                float(component_of[node]),
                float(clustering[node]),
                mean_neighbour_degree,
            ]
        )
    return np.asarray(rows, dtype=np.float32), node_index


def build_edge_index(graph: nx.Graph, node_index: dict[str, int]) -> np.ndarray:
    """COO edge index, both directions (the graph is undirected; PyG wants both)."""
    pairs = [(node_index[a], node_index[b]) for a, b in graph.edges()]
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64)
    forward = np.asarray(pairs, dtype=np.int64).T
    return np.concatenate([forward, forward[::-1]], axis=1)


def build_labels(node_index: dict[str, int], rings_ground_truth: dict[str, list[str]]) -> np.ndarray:
    """1 if the account belongs to an injected fraud ring, else 0."""
    ring_members = {account for members in rings_ground_truth.values() for account in members}
    labels = np.zeros(len(node_index), dtype=np.float32)
    for account, idx in node_index.items():
        if account in ring_members:
            labels[idx] = 1.0
    return labels


def temporal_node_split(
    node_index: dict[str, int],
    txns: pd.DataFrame,
    test_fraction: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Split nodes by each account's FIRST transaction time.

    Same reasoning as the point-risk model's chronological split, applied to nodes: an
    account whose activity begins after the boundary is one the model has never seen, so
    a ring that forms late is genuinely held out. A random node split would scatter a
    single ring's members across train and test and let the model recover the rest of a
    ring from the members it memorised — a leak that reads as excellent recall.
    """
    first_seen = txns.groupby("account_id")["timestamp"].min()
    times = pd.Series(
        {account: first_seen.get(account, pd.Timestamp.max) for account in node_index},
    ).sort_values()

    cutoff = int(len(times) * (1 - test_fraction))
    train_accounts = set(times.index[:cutoff])

    train_mask = np.zeros(len(node_index), dtype=bool)
    test_mask = np.zeros(len(node_index), dtype=bool)
    for account, idx in node_index.items():
        if account in train_accounts:
            train_mask[idx] = True
        else:
            test_mask[idx] = True
    return train_mask, test_mask


def build_model(in_channels: int, hidden_channels: int = 32):
    _require_torch_geometric()
    import torch
    from torch_geometric.nn import SAGEConv

    class RingSAGE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.head = torch.nn.Linear(hidden_channels, 1)

        def forward(self, x, edge_index):
            x = torch.relu(self.conv1(x, edge_index))
            x = torch.relu(self.conv2(x, edge_index))
            return self.head(x).squeeze(-1)

    return RingSAGE()


def train_gnn(
    features: np.ndarray,
    edge_index: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    epochs: int = 120,
    learning_rate: float = 0.01,
    random_seed: int = 1337,
) -> GNNTrainResult:
    """Full-batch transductive training. Message passing runs over the whole graph — a
    test node's neighbourhood is legitimately visible at inference in a fraud setting —
    but the loss is computed only on training nodes, so test labels never inform weights.
    """
    torch = _require_torch_geometric()
    from sklearn.metrics import average_precision_score, roc_auc_score

    torch.manual_seed(random_seed)

    x = torch.from_numpy(features)
    ei = torch.from_numpy(edge_index)
    y = torch.from_numpy(labels)
    train_t = torch.from_numpy(train_mask)

    model = build_model(in_channels=features.shape[1])

    n_pos = float(labels[train_mask].sum())
    n_neg = float(train_mask.sum() - n_pos)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0), dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    final_loss = float("nan")
    for epoch in range(epochs):
        model.train()
        optimiser.zero_grad()
        loss = criterion(model(x, ei)[train_t], y[train_t])
        loss.backward()
        optimiser.step()
        final_loss = float(loss.item())
        if (epoch + 1) % 30 == 0:
            print(f"  epoch {epoch + 1}/{epochs}  loss={final_loss:.4f}")

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(x, ei)).numpy()

    y_test = labels[test_mask]
    test_scores = scores[test_mask]
    # A held-out slice with no positives makes both metrics undefined; say so rather
    # than reporting a number that means nothing.
    if y_test.sum() == 0 or y_test.sum() == len(y_test):
        roc = pr = float("nan")
    else:
        roc = float(roc_auc_score(y_test, test_scores))
        pr = float(average_precision_score(y_test, test_scores))

    return GNNTrainResult(
        model=model,
        node_index={},
        roc_auc=roc,
        pr_auc=pr,
        n_train_nodes=int(train_mask.sum()),
        n_test_nodes=int(test_mask.sum()),
        epochs_run=epochs,
        final_loss=final_loss,
    )


def predict_ring_scores(model, features: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    torch = _require_torch_geometric()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(features), torch.from_numpy(edge_index))).numpy()
