import math
import numpy as np
import torch
import csv
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from utils import *
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T
import pandas as pd
from torch_geometric.utils import degree, to_networkx
from torch_geometric.datasets import TUDataset
import random
import json
import pickle
import sys,os
import traceback

from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import argparse
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
limits = plt.axis("off") 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='reddit_threads')
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--exp', type=str, default='nctefa', help="deg, nctefa, ac,concat")
parser.add_argument('--seed', type=int, default=789)
parser.add_argument('--model', type=str, default="GraphConv", help="Choose from: SAGEConv, GCNConv, ResGatedGraphConv, TransformerConv, GATConv")
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--echo_epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--early_stopping', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--dropout', type=float, default=0.5)
args = parser.parse_args()

# dataset is available here: https://github.com/benedekrozemberczki/datasets#twitch-ego-nets
path = "data/"
model_path = "params/"
res_path = "results/"

if not os.path.isdir(res_path):
    os.mkdir(res_path)
def logger(info):
    f = open(os.path.join(res_path, 'result.csv'), 'a')
    print(info, file=f)

################################PREPROCESSING DATA###############################################33
# path_to_data = f"./dataset/{args.dataset}/git_edges.json"
# path_to_target = f"./dataset/{args.dataset}/git_target.csv"


# dataset = json.load(open(path_to_data))
# targets = pd.read_csv(path_to_target)

# fix_seed(args.seed)

# graphs_unsorted, labels_unsorted = [],[]
# for k, edge_list in dataset.items():
#     index = int(k)
#     g = nx.Graph()
#     g.add_edges_from(edge_list)
#     y = targets['target'].iloc[index]
#     graphs_unsorted.append(g)
#     labels_unsorted.append(y)
#     if index%1000==0:
#         print(index)
# nodes = [g.number_of_nodes() for g in graphs_unsorted]
# sorted_data = sorted(zip(graphs_unsorted, labels_unsorted, nodes), key=lambda x: x[2])
# graphs, labels, sorted_nodes = zip(*sorted_data)

# print("len graphs:", len(graphs))
# b = 1000
# for i in range(0,len(graphs),b):
    
#     if (i<len(graphs)) and (b>len(graphs)):
#         batch = graphs[i:]
#         l = labels[i:]
#         indexes = [x for x in range(i,len(graphs))]
        
#     else:
#         batch = graphs[i:b]
#         l = labels[i:b]
#         indexes = [x for x in range(i,b)]
#     # print(len(batch),i, b)
#     # print("i:",i) 
#     print(len(batch), len(l), len(indexes))
#     # print(indexes[-5:])
#     # print(indexes[:-5])
#     b += 1000
#     # if i>11499:
#     #     break
#     num_jobs = -1
#     # # Parallel execution using joblib with explicit closing
#     with Parallel(n_jobs=num_jobs) as parallel:
#         parallel(
#             delayed(get_control_energy_gazers)(index, g, y) for (index, y, g) in zip(indexes,l, batch)
#         )


def one_hot_degree_encoding(data):
    max_degree = 0
    for d in data:
        row, col = d.edge_index
        degree = torch.zeros(d.num_nodes,dtype=torch.int)
        degree.scatter_add_(0, row, torch.ones_like(row, dtype=torch.int))
        
        max_deg = int(degree.max().item())
        if max_deg>max_degree:
            max_degree = max_deg
        
    new_dataset = []
    for d in data:
        x = torch.zeros((d.num_nodes, max_degree + 1),dtype = torch.float) 
        degrees = torch.zeros(d.num_nodes,dtype=torch.int)
        row, col = d.edge_index
        degrees.scatter_add_(0, row, torch.ones_like(row, dtype=torch.int)) 
        x[range(len(degrees)), degrees] = 1 
        new_data = Data(x=x, edge_index=d.edge_index, y=d.y)  # Create a new Data object
        new_dataset.append(new_data)
    return new_dataset

path_to_data = f"data/{args.dataset}/processed/"
files = os.listdir(path_to_data)

dataset = []
for f in files:
    with open(os.path.join(path_to_data, f), "rb") as file:
        inst = pickle.load(file)
    x = inst.get('x')  
    data = Data(
        x=torch.tensor(x, dtype=torch.float), 
        edge_index=from_networkx(inst.get("g")).edge_index,
        y=inst.get("y")
    )
    dataset.append(data)

if args.exp == 'deg':
    dataset = one_hot_degree_encoding(dataset)
else:
    for data in dataset:
        x = data.x.numpy() 
        if args.exp == 'nctefa':
            new_feat = x[:, :4]
        elif args.exp == 'ac':
            x_col = x[:, 0]
            hist, edges = np.histogram(x_col, bins=19)
            new_feat = []
            for d in x_col:
                bin_index = np.digitize(d, edges)
                xx = np.zeros(20)
                xx[bin_index - 1] = 1
                new_feat.append(xx)
            new_feat = np.array(new_feat)
        else:
            new_feat_list = []
            for i in range(x.shape[1]):
                col = x[:, i]
                hist, edges = np.histogram(col, bins=9)
                col_encoded = []
                for d in col:
                    bin_index = np.digitize(d, edges)
                    xx = np.zeros(10)
                    xx[bin_index - 1] = 1
                    col_encoded.append(xx)
                new_feat_list.append(np.array(col_encoded))
            new_feat = np.concatenate(new_feat_list, axis=1)
        
        data.x = torch.tensor(new_feat, dtype=torch.float)
    
print("dataset has been loaded!", len(dataset))

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(args.device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out.view(-1).cpu(), torch.tensor(data.y, dtype=torch.float))
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs

    return total_loss / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    y_pred, y_true = [], []
    for data in loader:
        data = data.to(args.device)
        logits = model(data.x, data.edge_index, data.batch)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(torch.tensor(data.y, dtype=torch.float))
    return roc_auc_score(torch.cat(y_true), torch.cat(y_pred))

labels = [d.y for d in dataset]
print("class distribution:", np.sum(labels), len(labels)-np.sum(labels))
num_features = dataset[0].x.shape[1]
num_folds = 10
stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=args.seed)
train_loss_global, test_roc_global, val_roc_global = [],[],[]

train_indices, test_indices = train_test_split(list(range(len(labels))),
    test_size=0.10, stratify=labels,random_state=args.seed,shuffle = True)

train_dataset = [dataset[i] for i in train_indices]
train_labels = [d.y.item() for d in train_dataset]
test_dataset = [dataset[i] for i in test_indices]
test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

for fold, (train_tmp, val_indices) in enumerate(stratified_kfold.split(range(len(train_dataset)), train_labels)):
    fix_seed(args.seed)
    print(f"\nFold {fold + 1}:")
    # tmp = [dataset[i] for i in train_tmp]
    train_data = [train_dataset[i] for i in train_tmp]
    val_dataset = [train_dataset[i] for i in val_indices]
    
    print("dataset {} loaded with train {} val {} test {} splits, num_features: {}".format(args.dataset,len(train_data), len(val_dataset), len(test_dataset),num_features))
    num_nodes = sorted([data.num_nodes for data in train_data])
    k = num_nodes[int(math.ceil(0.6 * len(num_nodes))) - 1]
    k = max(10, k)
    print("k:", k)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    
    print("dataset {} loaded with train {} val {} test {} splits, num_features: {}".format(args.dataset,len(train_dataset), len(val_dataset), len(test_dataset),num_features))
    num_nodes = sorted([data.num_nodes for data in train_dataset])
    k = num_nodes[int(math.ceil(0.6 * len(num_nodes))) - 1]
    k = max(10, k)
    print("k:", k)
    fix_seed(args.seed)

    # model = GCN(num_features, args.hidden, args.num_layers,args.model,k=k).to(args.device)
    model = GCN(num_features, args.hidden, args.num_layers, args.model, k=k).to(args.device)
    # print(model)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_val_roc,best_val_loss = 0.0,0.0
    for epoch in range(args.epochs):
        loss = train()
        val_roc = test(val_loader)
        test_roc = test(test_loader)
        
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, '
                f'val: {val_roc:.4f}, Test: {test_roc:.4f}')
        if val_roc > best_val_roc:
            best_val_roc = val_roc
            best_model_state = model.state_dict()
            # if epoch> int(args.epochs/2):
            torch.save(model.state_dict(), model_path + args.dataset+args.model+'task-checkpoint-best-roc_ctrl_encoded.pkl')
            # save_dir = os.path.join(model_path, args.dataset + args.model)
            # os.makedirs(save_dir, exist_ok=True) 
            # torch.save(model.state_dict(), os.path.join(save_dir, 'task-checkpoint-best-roc_ctrl_encoded.pkl'))

    model.load_state_dict(torch.load(model_path + args.dataset+args.model+'task-checkpoint-best-roc_ctrl_encoded.pkl'))
    model.eval()
    test_rocc = test(test_loader)
    val_rocc = test(val_loader)
    test_roc_global.append(test_rocc)
    val_roc_global.append(val_rocc)

log = f'dataset: {args.dataset}, experiment: {args.exp}, model: {args.model}, lr: {args.lr}, batch size:{args.batch_size}, hidden:{args.hidden}, folds: 10, val_roc: {np.mean(val_roc_global):.4f}, test_roc: {np.mean(test_roc_global):.4f}±{np.std(test_roc_global):.4f}'

print(log)

logger(log)