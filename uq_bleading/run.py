import math
import operator
from itertools import chain, product
from functools import partial
from pathlib import Path
from typing import Any, Optional, Callable, Tuple, Dict, Sequence, NamedTuple

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
from torch import Tensor, LongTensor

import sys

import torch_geometric
from torch_geometric.transforms import BaseTransform, Compose
from torch_geometric.datasets import QM9
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.nn.aggr import SumAggregation
import torch_geometric.nn as geom_nn
import torch.nn.functional as F
import torch.distributions as dist

import matplotlib as mpl
import matplotlib.pyplot as plt
from torch_scatter import scatter

HERE = Path(__file__).parent
DATA = HERE / "data"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(0)


def complete_edge_index(n: int) -> LongTensor:
    """
    Constructs a complete edge index.

    NOTE: representing complete graphs
    with sparse edge tensors is arguably a bad idea
    due to performance reasons, but for this tutorial it'll do.

    Parameters
    ----------
    n : int
        the number of nodes in the graph.

    Returns
    -------
    LongTensor
        A PyTorch `edge_index` represents a complete graph with n nodes,
        without self-loops. Shape (2, n).
    """
    # filter removes self loops
    edges = list(filter(lambda e: e[0] != e[1], product(range(n), range(n))))
    return torch.tensor(edges, dtype=torch.long).T

def add_complete_graph_edge_index(data: Data) -> Data:
    """
    On top of any edge information already there,
    add a second edge index that represents
    the complete graph corresponding to a  given
    torch geometric data object

    Parameters
    ----------
    data : Data
        The torch geometric data object.

    Returns
    -------
    Data
        The torch geometric `Data` object with a new
        attribute `complete_edge_index` as described above.
    """
    data.complete_edge_index = complete_edge_index(data.num_nodes)
    return data

class QM9DataModule:
    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        target_idx: int = 0,
        seed: float = 420,
    ) -> None:
        """Encapsulates everything related to the dataset

        Parameters
        ----------
        train_ratio : float, optional
            fraction of data used for training, by default 0.8
        val_ratio : float, optional
            fraction of data used for validation, by default 0.1
        test_ratio : float, optional
            fraction of data used for testing, by default 0.1
        target_idx : int, optional
            index of the target (see torch geometric docs), by default 5 (electronic spatial extent)
            (https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html?highlight=qm9#torch_geometric.datasets.QM9)
        seed : float, optional
            random seed for data split, by default 420
        """
        assert sum([train_ratio, val_ratio, test_ratio]) == 1
        self._dataset = self.dataset()
        self.target_idx = target_idx
        self.num_examples = len(self.dataset())
        rng = np.random.default_rng(seed)
        self.shuffled_index = rng.permutation(self.num_examples)
        self.train_split = self.shuffled_index[: int(self.num_examples * train_ratio)]
        self.val_split = self.shuffled_index[
            int(self.num_examples * train_ratio) : int(
                self.num_examples * (train_ratio + val_ratio)
            )
        ]
        self.test_split = self.shuffled_index[
            int(self.num_examples * (train_ratio + val_ratio)) : self.num_examples
        ]

        train_targets = self._dataset.data.y[self.train_split]
        self.y_mean = train_targets.mean()
        self.y_std = train_targets.std()

        self._dataset.transform = self.normalize_data

    def dataset(self, transform=None) -> QM9:
        dataset = QM9(DATA, pre_transform=add_complete_graph_edge_index, force_reload=True)

        dataset.data.y = dataset.data.y[:, self.target_idx].view(-1, 1)
        return dataset

    def normalize_data(self, data: Data) -> Data:
        data.y = (data.y - self.y_mean) / self.y_std
        return data

    def loader(self, split, **loader_kwargs) -> DataLoader:
        dataset = self._dataset()[split]
        return DataLoader(dataset, **loader_kwargs)

    def train_loader(self, **loader_kwargs) -> DataLoader:
        return self.loader(self.train_split, shuffle=True, **loader_kwargs)

    def val_loader(self, **loader_kwargs) -> DataLoader:
        return self.loader(self.val_split, shuffle=False, **loader_kwargs)

    def test_loader(self, **loader_kwargs) -> DataLoader:
        return self.loader(self.test_split, shuffle=False, **loader_kwargs)

data_module = QM9DataModule()


class NaiveEuclideanGNN(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_layers: int,
        num_spatial_dims: int,
        final_embedding_size: Optional[int] = None,
        act: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()
        # NOTE nn.Embedding acts like a lookup table.
        # Here we use it to store each atomic number in [0,100]
        # a learnable, fixed-size vector representation
        self.f_initial_embed = nn.Embedding(100, hidden_channels)
        self.f_pos_embed = nn.Linear(num_spatial_dims, hidden_channels)
        self.f_combine = nn.Sequential(nn.Linear(2 * hidden_channels, hidden_channels), act)

        if final_embedding_size is None:
            final_embedding_size = hidden_channels

        # Graph isomorphism network as main GNN
        # (see Talktorial 034)
        # takes care of message passing and
        # Learning node-level embeddings
        self.gnn = geom_nn.models.GIN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=final_embedding_size,
            num_layers=num_layers,
            act=act,
        )

        # modules required for aggregating node embeddings
        # into graph embeddings and making graph-level predictions
        self.aggregation = geom_nn.aggr.SumAggregation()
        self.f_predict_alpha = nn.Sequential(
            nn.Linear(final_embedding_size, final_embedding_size),
            act,
            nn.Linear(final_embedding_size, 1),
        )

        self.f_predict_beta = nn.Sequential(
            nn.Linear(final_embedding_size, final_embedding_size),
            act,
            nn.Linear(final_embedding_size, 1),
        )

        self.f_predict_nu = nn.Sequential(
            nn.Linear(final_embedding_size, final_embedding_size),
            act,
            nn.Linear(final_embedding_size, 1),
        )

        self.f_predict_gamma = nn.Sequential(
            nn.Linear(final_embedding_size, final_embedding_size),
            act,
            nn.Linear(final_embedding_size, 1),
        )

    def encode(self, data: Data) -> Tensor:
        # initial atomic number embedding and embedding od positional information
        atom_embedding = self.f_initial_embed(data.z)
        pos_embedding = self.f_pos_embed(data.pos)

        # treat both as plain node-level features and combine into initial node-level
        # embedddings
        initial_node_embed = self.f_combine(torch.cat((atom_embedding, pos_embedding), dim=-1))

        # message passing
        # NOTE in contrast to the EGNN implemented later, this model does use bond information
        # i.e., data.egde_index stems from the bond adjacency matrix
        node_embed = self.gnn(initial_node_embed, data.edge_index)
        return node_embed

    def forward(self, data: Data) -> Tensor:
        node_embed = self.encode(data)
        aggr = self.aggregation(node_embed, data.batch)
        nu = F.softplus(self.f_predict_nu(aggr))
        epsilon = 1e-4
        alpha = F.softplus(self.f_predict_alpha(aggr)) + 1
        alpha = torch.clamp(alpha, min=1.0 + epsilon)
        beta = F.softplus(self.f_predict_beta(aggr))
        gamma = self.f_predict_gamma(aggr)
        aleatoric_uncertainty = beta/(alpha-1)
        epistemic_uncertainty = beta/((alpha-1)*nu)
        return gamma, aleatoric_uncertainty, epistemic_uncertainty, nu, alpha, beta

class EquivariantMPLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        act: nn.Module,
    ) -> None:
        super().__init__()
        self.act = act
        self.residual_proj = nn.Linear(in_channels, hidden_channels, bias=False)

        # Messages will consist of two (source and target) node embeddings and a scalar distance
        message_input_size = 2 * in_channels + 1

        # equation (3) "phi_l" NN
        self.message_mlp = nn.Sequential(
            nn.Linear(message_input_size, hidden_channels),
            act,
        )
        # equation (4) "psi_l" NN
        self.node_update_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            act,
        )

    def node_message_function(
        self,
        source_node_embed: Tensor,  # h_i
        target_node_embed: Tensor,  # h_j
        node_dist: Tensor,  # d_ij
    ) -> Tensor:
        # implements equation (3)
        message_repr = torch.cat((source_node_embed, target_node_embed, node_dist), dim=-1)
        return self.message_mlp(message_repr)

    def compute_distances(self, node_pos: Tensor, edge_index: LongTensor) -> Tensor:
        row, col = edge_index
        xi, xj = node_pos[row], node_pos[col]
        # relative squared distance
        # implements equation (2) ||X_i - X_j||^2
        rsdist = (xi - xj).pow(2).sum(1, keepdim=True)
        return rsdist

    def forward(
        self,
        node_embed: Tensor,
        node_pos: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        row, col = edge_index
        dist = self.compute_distances(node_pos, edge_index)

        # compute messages "m_ij" from  equation (3)
        node_messages = self.node_message_function(node_embed[row], node_embed[col], dist)

        # message sum aggregation in equation (4)
        aggr_node_messages = scatter(node_messages, col, dim=0, reduce="sum")

        # compute new node embeddings "h_i^{l+1}"
        # (implements rest of equation (4))
        new_node_embed = self.residual_proj(node_embed) + self.node_update_mlp(
            torch.cat((node_embed, aggr_node_messages), dim=-1)
        )

        return new_node_embed


class EquivariantGNN(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        final_embedding_size: Optional[int] = None,
        target_size: int = 1,
        num_mp_layers: int = 2,
    ) -> None:
        super().__init__()
        if final_embedding_size is None:
            final_embedding_size = hidden_channels

        # non-linear activation func.
        # usually configurable, here we just use Relu for simplicity
        self.act = nn.ReLU()

        # equation (1) "psi_0"
        self.f_initial_embed = nn.Embedding(100, hidden_channels)

        # create stack of message passing layers
        self.message_passing_layers = nn.ModuleList()
        channels = [hidden_channels] * (num_mp_layers) + [final_embedding_size]
        for d_in, d_out in zip(channels[:-1], channels[1:]):
            layer = EquivariantMPLayer(d_in, d_out, self.act)
            self.message_passing_layers.append(layer)

        # modules required for readout of a graph-level
        # representation and graph-level property prediction
        self.aggregation = SumAggregation()
        self.f_predict_alpha = nn.Sequential(
            nn.Linear(final_embedding_size, final_embedding_size),
            self.act,
            nn.Linear(final_embedding_size, target_size),
        )

        self.f_predict_beta = nn.Sequential(
            nn.Linear(final_embedding_size, final_embedding_size),
            self.act,
            nn.Linear(final_embedding_size, target_size),
        )

        self.f_predict_nu = nn.Sequential(
            nn.Linear(final_embedding_size, final_embedding_size),
            self.act,
            nn.Linear(final_embedding_size, target_size),
        )

        self.f_predict_gamma = nn.Sequential(
            nn.Linear(final_embedding_size, final_embedding_size),
            self.act,
            nn.Linear(final_embedding_size, target_size),
        )

    def encode(self, data: Data) -> Tensor:
        # theory, equation (1)
        node_embed = self.f_initial_embed(data.z)
        # message passing
        # theory, equation (3-4)
        for mp_layer in self.message_passing_layers:
            # NOTE here we use the complete edge index defined by the transform earlier on
            # to implement the sum over $j \neq i$ in equation (4)
            node_embed = mp_layer(node_embed, data.pos, data.complete_edge_index)
        return node_embed

    def _predict(self, node_embed, batch_index) -> Tensor:
        aggr = self.aggregation(node_embed, batch_index)
        gamma = self.f_predict_gamma(aggr)
        nu = F.softplus(self.f_predict_nu(aggr))
        epsilon = 1e-4
        alpha = F.softplus(self.f_predict_alpha(aggr)) + 1
        alpha = torch.clamp(alpha, min=1.0 + epsilon)
        beta = F.softplus(self.f_predict_beta(aggr))

        aleatoric_uncertainty = beta/(alpha-1)
        epistemic_uncertainty = beta/((alpha-1)*nu)

        return gamma, aleatoric_uncertainty, epistemic_uncertainty, nu, alpha, beta

    def forward(self, data: Data) -> Tensor:
        node_embed = self.encode(data)
        pred = self._predict(node_embed, data.batch)
        return pred

# We will be using mean absolute error
# as a metric for validation and testing
def total_absolute_error(pred: Tensor, target: Tensor, batch_dim: int = 0) -> Tensor:
    """Total absolute error, i.e. sums over batch dimension.

    Parameters
    ----------
    pred : Tensor
        batch of model predictions
    target : Tensor
        batch of ground truth / target values
    batch_dim : int, optional
        dimension that indexes batch elements, by default 0

    Returns
    -------
    Tensor
        total absolute error
    """
    return (pred - target).abs().sum(batch_dim)

def tilted_loss(q, e):
    return torch.max(q * e, (q - 1) * e)

def np_tilted_loss(q, e):
    return np.maximum(q * e, (q - 1) * e)


def NIG_NLL(y, gamma, v, alpha, beta, w_i_dis, quantile, reduce=True):
    tau_two = 2.0 / (quantile * (1.0 - quantile))  # Scalar
    twoBlambda = 2.0 * 2.0 * beta * (1.0 + tau_two * w_i_dis.mean(dim=1, keepdim=True) * v)  # Shape: [batch_size, dimension]
    nll = (
        0.5 * torch.log(np.pi / v)  # Shape: [batch_size, dimension]
        - alpha * torch.log(twoBlambda)  # Shape: [batch_size, dimension]
        + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + twoBlambda)  # Shape: [batch_size, dimension]
        + torch.lgamma(alpha)  # Shape: [batch_size, dimension]
        - torch.lgamma(alpha + 0.5)  # Shape: [batch_size, dimension]
    )
    per_element_loss = nll.sum(dim=1)  # Sum over dimensions for each batch element
    return per_element_loss.mean() if reduce else per_element_loss


def KL_NIG(gamma, v, alpha, beta, gamma_p, omega_p, v_p, beta_p):
    raise NotImplementedError("KL_NIG function is not implemented")

def NIG_Reg(y, gamma, v, alpha, beta, w_i_dis, quantile, omega=0.01, reduce=True, kl=False):
    error = tilted_loss(quantile, y - gamma)  # Shape: [batch_size, dimension]
    w = abs(quantile - 0.5)  # Scalar weight based on quantile
    if kl:
        kl_div = KL_NIG(
            gamma, v, alpha, beta,
            gamma, omega, 1 + omega, beta
        )  # Shape: [batch_size, dimension]
        reg = error * kl_div  # Shape: [batch_size, dimension]
    else:
        evi = 2 * v + alpha + 1 / beta  # Shape: [batch_size, dimension]
        reg = error * evi  # Shape: [batch_size, dimension]
    per_element_loss = reg.sum(dim=1)  # Sum over dimensions for each batch element
    return per_element_loss.mean() if reduce else per_element_loss

def quant_evi_loss(y_true, gamma, v, alpha, beta, quantile, coeff=0.1, reduce=True):
    theta = (1.0 - 2.0 * quantile) / (quantile * (1.0 - quantile))  # Scalar
    mean_ = beta / (alpha - 1)  # Shape: [batch_size, dimension]
    rate = 1 / (mean_ + 1e-8)
    if torch.any(rate <= 0):
        print("Found non-positive rate values in Exponential distribution")
        print("rate min:", rate.min())
        print("rate max:", rate.max())
        print("mean_ min:", mean_.min())
        print("mean_ max:", mean_.max())
        print("alpha max:", alpha.max())
        print("alpha min:", alpha.min())
        print("beta max:", beta.max())
        print("beta min:", beta.min())

    if torch.any(mean_ <= 1e-8):
        print("Warning: Clipping mean_ values to 1e-8 to avoid invalid rates in the exponential distribution.")
    max_mean_value = 1e10
    mean_ = torch.clamp(mean_, min=1e-8, max=max_mean_value)
    exp_dist = dist.Exponential(rate=1 / mean_ )  # Shape: [batch_size, dimension]
    w_i_dis = exp_dist.sample().to(device)  # Shape: [batch_size, dimension]
    mu = gamma + theta * w_i_dis.mean(dim=1, keepdim=True)  # Shape: [batch_size, dimension]

    loss_nll = NIG_NLL(y_true, mu, v, alpha, beta, w_i_dis, quantile, reduce=reduce)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta, w_i_dis, quantile, reduce=reduce)

    return loss_nll + coeff * loss_reg

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: Callable[
    [Tensor, Tensor, Tensor, Tensor, Tensor, float, Optional[float], Optional[bool]], Tensor],
    pbar: Optional[Any] = None,
    optim: Optional[torch.optim.Optimizer] = None,
    verbose: bool = False,
):
    """Run a single epoch.

    Parameters
    ----------
    model : nn.Module
        the NN used for regression
    loader : DataLoader
        an iterable over data batches
    criterion : Callable[[Tensor, Tensor], Tensor]
        a criterion (loss) that is optimized
    pbar : Optional[Any], optional
        a tqdm progress bar, by default None
    optim : Optional[torch.optim.Optimizer], optional
        a optimizer that is optimizing the criterion, by default None
    """

    def step(
        data_batch: Data,
    ) -> Tuple[float, float]:
        """Perform a single train/val step on a data batch.

        Parameters
        ----------
        data_batch : Data

        Returns
        -------
        Tuple[float, float]
            Loss (mean squared error) and validation critierion (absolute error).
        """
        data_batch = data_batch.to(device)
        gamma, aleatoric_uncertainty, epistemic_uncertainty, nu, alpha, beta = model.forward(data_batch)
        if verbose:
            print("gamma", gamma)
            print("aleatoric_uncertainty", aleatoric_uncertainty)
            print("epistemic_uncertainty", epistemic_uncertainty)
            print("nu", nu)
            print("alpha", alpha)
            print("beta", beta)
        target = data_batch.y
        loss = criterion(target, gamma, nu, alpha, beta, 0.5)
        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        if verbose:
            print("loss", loss)
            total_absolute_error(gamma, target)
            print("total_absolute_error", total_absolute_error(gamma, target))
        return loss.detach().item(), total_absolute_error(gamma.detach(), target.detach())

    if optim is not None:
        model.train()
        # This enables pytorch autodiff s.t. we can compute gradients
        model.requires_grad_(True)
    else:
        model.eval()
        # disable autodiff: when evaluating we do not need to track gradients
        model.requires_grad_(False)

    total_loss = 0
    total_mae = 0
    for data in loader:
        data = data.to(device)
        loss, mae = step(data)
        total_loss += loss * data.num_graphs
        total_mae += mae
        if pbar is not None:
            pbar.update(1)

    return total_loss / len(loader.dataset), total_mae / len(loader.dataset)


def train_model(
    data_module: QM9DataModule,
    model: nn.Module,
    num_epochs: int = 30,
    lr: float = 3e-6,
    batch_size: int = 32,
    weight_decay: float = 1e-8,
    best_model_path: Path = DATA.joinpath("trained_model.pth"),
    verbose: bool = True,
) -> Dict[str, Any]:
    """Takes data and model as input and runs training, collecting additional validation metrics
    while doing so.

    Parameters
    ----------
    data_module : QM9DataModule
        a data module as defined earlier
    model : nn.Module
        a gnn model
    num_epochs : int, optional
        number of epochs to train for, by default 30
    lr : float, optional
        "learning rate": optimizer SGD step size, by default 3e-4
    batch_size : int, optional
        number of examples used for one training step, by default 32
    weight_decay : float, optional
        L2 regularization parameter, by default 1e-8
    best_model_path : Path, optional
        path where the model weights with lowest val. error should be stored
        , by default DATA.joinpath("trained_model.pth")

    Returns
    -------
    Dict[str, Any]
        a training result, ie statistics and info about the model
    """
    # create data loaders
    train_loader = data_module.train_loader(batch_size=batch_size)
    val_loader = data_module.val_loader(batch_size=batch_size)

    # setup optimizer and loss
    optim = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-8)
    loss_fn = quant_evi_loss

    # keep track of the epoch with the best validation mae
    # st we can save the "best" model weights
    best_val_mae = float("inf")

    # Statistics that will be plotted later on
    # and model info
    result = {
        "model": model,
        "path_to_best_model": best_model_path,
        "train_loss": np.full(num_epochs, float("nan")),
        "val_loss": np.full(num_epochs, float("nan")),
        "train_mae": np.full(num_epochs, float("nan")),
        "val_mae": np.full(num_epochs, float("nan")),
    }

    def update_statistics(i_epoch: int, **kwargs: float):
        for key, value in kwargs.items():
            result[key][i_epoch] = value

    def desc(i_epoch: int) -> str:
        return " | ".join(
            [f"Epoch {i_epoch + 1:3d} / {num_epochs}"]
            + [
                f"{key}: {value[i_epoch]:8.2f}"
                for key, value in result.items()
                if isinstance(value, np.ndarray)
            ]
        )

    # main training loop
    for i_epoch in range(0, num_epochs):
        progress_bar = tqdm(total=len(train_loader) + len(val_loader))
        try:
            # tqdm for reporting progress
            progress_bar.set_description(desc(i_epoch))

            # training epoch
            train_loss, train_mae = run_epoch(model, train_loader, loss_fn, progress_bar, optim, verbose)
            # validation epoch
            val_loss, val_mae = run_epoch(model, val_loader, loss_fn, progress_bar, verbose=verbose)

            update_statistics(
                i_epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_mae=train_mae,
                val_mae=val_mae,
            )

            progress_bar.set_description(desc(i_epoch))

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), best_model_path)
        finally:
            progress_bar.close()

    return result

@torch.no_grad()
def test_model(model: nn.Module, data_module: QM9DataModule, model_name) -> Tuple[float, Tensor, Tensor]:
    """
    Test a model.

    Parameters
    ----------
    model : nn.Module
        a trained model
    data_module : QM9DataModule
        a data module as defined earlier
        from which we'll get the test data

    Returns
    -------
    _Tuple[float, Tensor, Tensor]
        Test MAE, and model predictions & targets for further processing
    """
    test_mae = 0
    preds, targets = [], []
    loader = data_module.test_loader()
    enum = 0
    aleatoric_uqs, epistemic_uqs = [], []
    for data in loader:
        data = data.to(device)
        pred = model(data)[0]
        target = data.y
        preds.append(pred)
        targets.append(target)
        test_mae += total_absolute_error(pred, target).item()
        gamma, aleatoric_uncertainty, epistemic_uncertainty, nu, alpha, beta = model.forward(data)
        aleatoric_uqs.append(aleatoric_uncertainty)
        epistemic_uqs.append(epistemic_uncertainty)
        gamma = gamma.detach().cpu()
        aleatoric_uncertainty = aleatoric_uncertainty.detach().cpu()
        aleatoric_uncertainty = aleatoric_uncertainty.numpy()
        epistemic_uncertainty = epistemic_uncertainty.detach().cpu()
        epistemic_uncertainty = epistemic_uncertainty.numpy()
        nu = nu.detach().cpu()
        nu = nu.numpy()
        alpha = alpha.detach().cpu()
        alpha = alpha.numpy()
        beta = beta.detach().cpu()
        beta = beta.numpy()
        y = data.y.detach().cpu()
        y = y.numpy()
        enum += 1


    test_mae = test_mae / len(data_module.test_split)
    preds_cpu = [p.detach().cpu() for p in preds]
    targets_cpu = [t.detach().cpu() for t in targets]
    aleatoric_uqs_cpu = [a.detach().cpu() for a in aleatoric_uqs]
    epistemic_uqs_cpu = [e.detach().cpu() for e in epistemic_uqs]
    np.save(f'preds_{model_name}.npy', np.array(preds_cpu))
    np.save(f'targets_{model_name}.npy', np.array(targets_cpu))
    np.save(f'aleatoric_uqs_{model_name}.npy', np.array(aleatoric_uqs_cpu))
    np.save(f'epistemic_uqs_{model_name}.npy', np.array(epistemic_uqs_cpu))



    return test_mae, torch.cat(preds, dim=0), torch.cat(targets, dim=0)

model = EquivariantGNN(hidden_channels=64, num_mp_layers=2).to(device)

egnn_train_result = train_model(
    data_module,
    model,
    num_epochs=25,
    lr=3e-6,
    batch_size=32,
    weight_decay=1e-8,
    best_model_path=DATA.joinpath("trained_egnn.pth"),
)

gcn_baseline = NaiveEuclideanGNN(64, 4, 3).to(device)

gcn_train_result = train_model(
    data_module,
    gcn_baseline,
    num_epochs=100,
    lr=3e-6,
    batch_size=32,
    best_model_path=DATA.joinpath("trained_gnn.pth"),
)

gcn_num_params = sum(p.numel() for p in gcn_train_result["model"].parameters())
egnn_num_params = sum(p.numel() for p in egnn_train_result["model"].parameters())

for key, value in {"GCN": gcn_num_params, "EGNN": egnn_num_params}.items():
    print(f"{key} has {value} parameters")

gcn_model = gcn_train_result["model"]
gcn_model.load_state_dict(torch.load(gcn_train_result["path_to_best_model"], map_location=device))
gcn_test_mae, gcn_preds, gcn_targets = test_model(gcn_model, data_module, 'gnn')

egnn_model = egnn_train_result["model"]
egnn_model.load_state_dict(torch.load(egnn_train_result["path_to_best_model"], map_location=device))
egnn_test_mae, egnn_preds, egnn_targets = test_model(egnn_model, data_module, 'egnn')

print(f"EGNN test MAE: {egnn_test_mae}")
print(f"GNN test MAE: {gcn_test_mae}")

