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

def load_clean_data(fname):
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
        d.x = torch.tensor(d.x.astype("float32"), dtype=torch.float32)
        d.y=torch.tensor(d.y.astype("float32"), dtype=torch.float32)
        d.edge_attr=torch.tensor(d.edge_attr, dtype=torch.float32)
        d.edge_index=torch.tensor(d.edge_index, dtype=torch.long)
        del d.smile
        dat_new = dat_new+[d]
    return dat_new

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


if __name__=='__main__':
    # Input arguments
    in_fname = sys.argv[1] # pytorch graphs file
    seed(0)

    # clean up data and split into train/val/test
    data=load_clean_data(in_fname)
    
    # Test data = last 10 percent of suffled data
    random.shuffle(data)
    test_loader = DataLoader(data[m.floor(0.9*len(data)):], batch_size=32, shuffle=False)

    # Setup model architecture
    #model = MPNNModel(num_layers=4, emb_dim=64, in_dim=6, edge_dim=1, out_dim=1)
    model=torch.load("GNN_Best.pt", weights_only=False)
    model_name = type(model).__name__
    test_error, pred, act, feat = eval(model, test_loader, device="cuda")
    print("Test Error: ", test_error)
    df = pd.DataFrame({"Pred": np.array(pred), "Act": np.array(act) })
    df["Error"] = df["Pred"]- df["Act"]
    df["Abs Error"] = df["Error"].abs()
    print("Number of data: ", len(df))
    print(df.sort_values(by="Abs Error", ascending=False).head(20))
