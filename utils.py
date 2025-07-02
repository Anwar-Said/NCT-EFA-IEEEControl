
from torch.nn import BCEWithLogitsLoss, Conv1d, MaxPool1d, ModuleList
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, GraphConv,SAGEConv,GCNConv,ResGatedGraphConv,TransformerConv,GATConv, global_sort_pool
import torch
import math
from nctpy.utils import matrix_normalization
from nctpy.metrics import ave_control
import networkx as nx
import numpy as np
import random,os
from torch_geometric.utils import from_networkx
import pickle
import traceback
from nctpy.energies import gramian
from nctpy.utils import matrix_normalization


def get_weighted_adj(gramian, adj, num_bins):
    weights = np.zeros((adj.shape[0], adj.shape[1]), dtype= float)
    for i, row in enumerate(gramian):
        hist, bin_edges = np.histogram(row, num_bins)
        for j, r in enumerate(row):
            bin_index = np.digitize(r, bin_edges, right=False)
            weights[i,j] = bin_index
    return weights*adj

def get_control_energy(g,y):
    try:
        if not nx.is_connected(g):
            largest_cc = max(nx.connected_components(g), key=len)
            g = g.subgraph(largest_cc)
        adj = nx.to_numpy_array(g)
        system = 'continuous'
        adj= matrix_normalization(adj, c=1, system=system)
        avg_control = ave_control(A_norm=adj, system=system)
        # print(avg_control)
        closeness_cent =  list(nx.closeness_centrality(g).values())
        # print(closeness_cent)
        bet_centrality = list(nx.betweenness_centrality(g).values())
        eig_centrality = list(nx.eigenvector_centrality(g).values())
        feat = np.stack([avg_control,closeness_cent,bet_centrality,eig_centrality],axis=1)
       
    except:
        feat = np.random.rand(g.number_of_nodes(), 4)
    data = Data(x = torch.tensor(feat,dtype=torch.float),edge_index = from_networkx(g).edge_index, y= y)
   
    return data

def get_control_encoding(path_to_data,index,f,bins):
    file_path = "data/github_stargazers/processed_weighted/"+str(index)+".pkl"
    if not os.path.exists(file_path):
        try:
            with open(os.path.join(path_to_data,f), "rb") as file:
                    inst = pickle.load(file)
            g = inst.get("g")
            A = nx.to_numpy_array(g)
            A_norm = matrix_normalization(A=A, c=1, system="continuous")
            gram = gramian(A_norm, T=1, system= 'continuous') 
            weighted_adj = torch.tensor(get_weighted_adj(gram, A, num_bins=bins), dtype=float)
            edge_index = (weighted_adj > 0).nonzero().t()
            row, col = edge_index
            edge_weight = weighted_adj[row, col]
            data = Data(x = torch.tensor(inst.get('x'),dtype=torch.float),edge_index = edge_index,edge_weight=edge_weight, y= inst.get("y"))
            with open("data/github_stargazers/processed_weighted/"+str(index)+".pkl", "wb") as file:
                pickle.dump(data, file)
            print("index {} saved!".format(index))
        except:
            print("exception occured") 

def get_control_energy_gazers(index,g,y):
    file_path = "data/github_stargazers/processed/"+str(index)+".pkl"
    if not os.path.exists(file_path):
        try:
            print(file_path)
            if not nx.is_connected(g):
                largest_cc = max(nx.connected_components(g), key=len)
                g = g.subgraph(largest_cc)
            adj = nx.to_numpy_array(g)
            system = 'continuous'
            adj= matrix_normalization(adj, c=1, system=system)
            avg_control = ave_control(A_norm=adj, system=system)
            # print(avg_control)
            closeness_cent =  list(nx.closeness_centrality(g).values())
            # print(closeness_cent)
            bet_centrality = list(nx.betweenness_centrality(g).values())
            eig_centrality = list(nx.eigenvector_centrality(g).values())
            # node degree 
            node_degree = list(dict(g.degree()).values())
            feat = np.stack([avg_control,closeness_cent,bet_centrality,eig_centrality,node_degree],axis=1)
        
        except Exception as e:
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {e}")

            traceback.print_exc()
            
            feat = np.random.rand(g.number_of_nodes(), 5)
            print("index {} saved with exception!".format(index))

        data = {"index":index,
                "x": feat,
                "g":g,
                "y":y}
        with open("data/github_stargazers/processed/"+str(index)+".pkl", "wb") as file:
            pickle.dump(data, file)
        print("index {} saved!".format(index))
    
    # data = Data(x = torch.tensor(feat,dtype=torch.float),edge_index = from_networkx(g).edge_index, y= y)

    # return data
def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
class GCN(torch.nn.Module):
    def __init__(self,num_features, hidden_channels,num_layers,gnn,k):
        super().__init__()
        self.k = k
        GNN = eval(gnn)

        self.convs = ModuleList()
        self.convs.append(GNN(num_features, hidden_channels))
        for i in range(0, num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))
        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.mlp = MLP([dense_dim, 32, 1], dropout=0.5, batch_norm=False)

    def forward(self, x, edge_index, batch):
        xs = [x]
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]
        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = self.conv1(x).relu()
        x = self.maxpool1d(x)
        x = self.conv2(x).relu()
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]
        return self.mlp(x)
class GCN_weighted(torch.nn.Module):
    def __init__(self,num_features, hidden_channels,num_layers,gnn,k):
        super().__init__()
        self.k = k
        GNN = eval(gnn)

        self.convs = ModuleList()
        self.convs.append(GNN(num_features, hidden_channels))
        for i in range(0, num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))
        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.mlp = MLP([dense_dim, 32, 1], dropout=0.5, batch_norm=False)

    def forward(self, x, edge_index,edge_weights, batch):
        xs = [x]
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index, edge_weights).tanh()]
        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = self.conv1(x).relu()
        x = self.maxpool1d(x)
        x = self.conv2(x).relu()
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]
        return self.mlp(x)