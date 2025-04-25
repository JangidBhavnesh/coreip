import os
import sys
import time
import random
import numpy as np
import pandas as pd
import math as m

import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential

import torch_geometric
from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter
import warnings
warnings.filterwarnings('ignore')


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def load_clean_data(fname, device):
    dat = torch.load(fname, weights_only=False)
    dat_new = []
    for d in dat:
        bad_smiles = ['[2H]C([2H])Br',
 '[2H]C([2H])([2H])C(=O)C([2H])([2H])[2H]',
 '[H-].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[Co]',
 '[H-].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[Mn]',
 '[CH2]O.[CH2]O.[CH2]O.[CH2]O.[CH2]O.[CH2]O.[CH2]O.[CH2]O.[CH2]O.[CH]O.[CH]O.[CH]O.[Co].[Co].[Co].[Fe].[H]']
        if d.smile in bad_smiles:
            print("Bad Smile Found:", d.smile)
            del d
            continue
        d.x = torch.tensor(d.x.astype("float32"), dtype=torch.float32).to(device)
        d.y=torch.tensor(d.y.astype("float32"), dtype=torch.float32).to(device)
        d.edge_attr=torch.tensor(d.edge_attr, dtype=torch.float32).to(device)
        d.edge_index=torch.tensor(d.edge_index, dtype=torch.long).to(device)
        del d.smile
        dat_new = dat_new+[d]
    return dat_new

def split_data(data):
    print("\nNumber of Graphs: ", len(data))

    random.shuffle(data)
    train_dataset = data[:m.floor(0.8*len(data))]
    val_dataset = data[m.floor(0.8*len(data)):m.floor(0.9*len(data))]
    test_dataset = data[m.floor(0.9*len(data)):]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"Datset splits: {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples.")
    return train_loader, val_loader, test_loader


class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=1, aggr='add'):
        """Message Passing Neural Network Layer

        Args:
            emb_dim: (int) - hidden dimension `d`
            edge_dim: (int) - edge feature dimension `d_e`
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        # MLP `\psi` for computing messages `m_ij`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: (2d + d_e) -> d
        self.mlp_msg = Sequential(
            Linear(2*emb_dim + edge_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
          )

        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: 2d -> d
        self.mlp_upd = Sequential(
            Linear(2*emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
          )

    def forward(self, h, edge_index, edge_attr):
        """
        The forward pass updates node features `h` via one round of message passing.

        As our MPNNLayer class inherits from the PyG MessagePassing parent class,
        we simply need to call the `propagate()` function which starts the
        message passing procedure: `message()` -> `aggregate()` -> `update()`.

        The MessagePassing class handles most of the logic for the implementation.
        To build custom GNNs, we only need to define our own `message()`,
        `aggregate()`, and `update()` functions (defined subsequently).

        Args:
            h: (n, d) - initial node features
            edge_index: (e, 2) - pairs of edges (i, j)
            edge_attr: (e, d_e) - edge features

        Returns:
            out: (n, d) - updated node features
        """
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, edge_attr):
        """Step (1) Message

        The `message()` function constructs messages from source nodes j
        to destination nodes i for each edge (i, j) in `edge_index`.

        The arguments can be a bit tricky to understand: `message()` can take
        any arguments that were initially passed to `propagate`. Additionally,
        we can differentiate destination nodes and source nodes by appending
        `_i` or `_j` to the variable name, e.g. for the node features `h`, we
        can use `h_i` and `h_j`.

        This part is critical to understand as the `message()` function
        constructs messages for each edge in the graph. The indexing of the
        original node features `h` (or other node variables) is handled under
        the hood by PyG.

        Args:
            h_i: (e, d) - destination node features
            h_j: (e, d) - source node features
            edge_attr: (e, d_e) - edge features

        Returns:
            msg: (e, d) - messages `m_ij` passed through MLP `\psi`
        """
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.mlp_msg(msg)

    def aggregate(self, inputs, index):
        """Step (2) Aggregate

        The `aggregate` function aggregates the messages from neighboring nodes,
        according to the chosen aggregation function ('sum' by default).

        Args:
            inputs: (e, d) - messages `m_ij` from destination to source nodes
            index: (e, 1) - list of source nodes for each edge/message in `input`

        Returns:
            aggr_out: (n, d) - aggregated messages `m_i`
        """
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

    def update(self, aggr_out, h):
        """
        Step (3) Update

        The `update()` function computes the final node features by combining the
        aggregated messages with the initial node features.

        `update()` takes the first argument `aggr_out`, the result of `aggregate()`,
        as well as any optional arguments that were initially passed to
        `propagate()`. E.g. in this case, we additionally pass `h`.

        Args:
            aggr_out: (n, d) - aggregated messages `m_i`
            h: (n, d) - initial node features

        Returns:
            upd_out: (n, d) - updated node features passed through MLP `\phi`
        """
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')

class MPNNModel(Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=6, edge_dim=1, out_dim=1):
        """Message Passing Neural Network model for graph property prediction

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
            out_dim: (int) - output dimension
        """
        super().__init__()

        # Linear projection for initial node features
        # dim: d_n -> d
        self.lin_in = Linear(in_dim, emb_dim)

        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))

        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        self.pool = global_mean_pool

        # Linear prediction head
        # dim: d -> out_dim
        self.lin_pred = Linear(emb_dim, out_dim)

    def forward(self, data):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns:
            out: (target_size, out_dim) - prediction for each node
        """
        h = self.lin_in(data.x) # (n, d_n) -> (n, d)
        for conv in self.convs:
            h = h + conv(h, data.edge_index, data.edge_attr) # (n, d) -> (n, d)

        out = self.lin_pred(h) # (target_size, d) -> (target_size, 1)
       
        return out.view(-1)


def train(model, train_loader, optimizer, device):
    model.train()
    loss_all = 0
    len_all = 0
    for data in train_loader:
        data=data.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = F.l1_loss(pred, data.y)
        loss.backward()
        loss_all += loss.item() * len(pred)
        len_all+=len(pred)
        optimizer.step()
    return loss_all/len_all

#evaluate
def eval(model, loader, device):
    model.eval()
    error=0
    len_all=0
    pred_collected = []
    act_collected = []
    feat_collected = []
    
    for data in loader:
        with torch.no_grad():
            pred = model(data)
            mse=F.l1_loss(pred,data.y)
            error+=mse.item()*len(pred)
            len_all+=len(pred)
            
            pred_collected = pred_collected + list(pred)
            act_collected = act_collected + list(data.y)
            feat_collected = feat_collected + list(data)
            
    return error/len_all, pred_collected, act_collected, feat_collected


def run_expt(model, model_name, train_loader, val_loader, test_loader, n_epochs=10):
    print(f"Running experiment for {model_name}, training on {len(train_loader.dataset)} samples for {n_epochs} epochs.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\nModel architecture:")
    print(model)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'Total parameters: {total_param}')
    model = model.to(device)

    # Adam optimizer with LR 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # LR scheduler which decays LR when validation metric doesn't improve
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=5, min_lr=0.00001)

    print("\nStart training:")
    best_val_error = None
    perf_per_epoch = [] # Track Test/Val MAE vs. epoch (for plotting)
    t = time.time()
    for epoch in range(1, n_epochs+1):
        # Call LR scheduler at start of each epoch
        lr = scheduler.optimizer.param_groups[0]['lr']

        # Train model for one epoch, return avg. training loss
        loss = train(model, train_loader, optimizer, device)

        # Evaluate model on validation set
        val_error, pred, act, feat = eval(model, val_loader, device)

        if best_val_error is None or val_error <= best_val_error:
            # Evaluate model on test set if validation metric improves
            test_error, pred_tst, act_tst, feat_collected = eval(model, test_loader, device)
            best_val_error = val_error

        if epoch % 5 == 0:
            # Print and track stats every 5 epochs
            print(f'Epoch: {epoch:03d}, LR: {lr:5f}, Loss: {loss:.7f}, '
                  f'Val MAE: {val_error:.7f}, Test MAE: {test_error:.7f}')

        scheduler.step(val_error)
        perf_per_epoch.append((loss, test_error, val_error, epoch, model_name))

    t = time.time() - t
    train_time = t/60
    print(f"\nDone! Training took {train_time:.2f} mins. Best validation MAE: {best_val_error:.7f}, corresponding test MAE: {test_error:.7f}.")
    torch.save(model, "GNN0.pt")
    return loss, best_val_error, test_error, train_time, perf_per_epoch, pred_tst, act_tst, feat_collected
    

if __name__=='__main__':
    # Input arguments
    in_fname = sys.argv[1] # pytorch graphs file
    seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Clean up data and split into train/val/test
    data=load_clean_data(in_fname, device)
    train_loader, val_loader, test_loader = split_data(data) 

    RESULTS = {}
    DF_RESULTS = pd.DataFrame(columns=["Train MAE", "Test MAE", "Val MAE", "Epoch", "Model"])

    # Setup model architecture
    model = MPNNModel(num_layers=4, emb_dim=64, in_dim=6, edge_dim=1, out_dim=1)
    model_name = type(model).__name__

    # Run train/validate/test
    loss, best_val_error, test_error, train_time, perf_per_epoch, pred, act, feat = run_expt(
    model,
    model_name,
    train_loader,
    val_loader,
    test_loader,
    n_epochs=100
)

    RESULTS[model_name] = (loss, best_val_error, test_error, train_time)
    df_temp = pd.DataFrame(perf_per_epoch, columns=["Train MAE", "Test MAE", "Val MAE", "Epoch", "Model"])
    DF_RESULTS = DF_RESULTS.append(df_temp, ignore_index=True)
    DF_RESULTS.to_csv("GNN0_Training_Results.csv")
