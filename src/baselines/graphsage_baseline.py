#!/usr/bin/env python3
import torch, time
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch.serialization import safe_globals
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

DATA_PATH = "/app/juliet_cepg_ast.pt"
with safe_globals([Data]):
    graphs = torch.load(DATA_PATH, weights_only=False)

train_split = int(0.7*len(graphs)); val_split=int(0.85*len(graphs))
train_graphs, val_graphs, test_graphs = graphs[:train_split], graphs[train_split:val_split], graphs[val_split:]
train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_graphs, batch_size=8)
test_loader  = DataLoader(test_graphs, batch_size=8)

in_dim = graphs[0].x.size(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.lin   = torch.nn.Linear(hidden, 1)
    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        g = global_mean_pool(x, batch)
        return self.lin(g).squeeze()

model = GraphSAGE(in_dim).to(device)
opt = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
loss_fn = torch.nn.BCEWithLogitsLoss()

def evaluate(loader):
    model.eval(); probs, labels = [], []
    for b in loader:
        b = b.to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(b.x, b.edge_index, b.batch)).cpu().numpy()
        probs.extend(p); labels.extend(b.y.cpu().numpy())
    return f1_score(labels, np.round(probs)), roc_auc_score(labels, probs)

for epoch in range(5):
    model.train(); total = 0
    for b in train_loader:
        b = b.to(device)
        out = model(b.x, b.edge_index, b.batch)
        loss = loss_fn(out, b.y.float())
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    f1, auc = evaluate(val_loader)
    print(f"Epoch {epoch:02d} | Loss={total/len(train_loader):.3f} | Val F1={f1:.3f} | AUC={auc:.3f}")

start = time.time(); f1, auc = evaluate(test_loader); latency = (time.time()-start)/len(test_graphs)*1000
print(f"📊 GraphSAGE → F1={f1:.3f} | AUC={auc:.3f} | Latency={latency:.2f} ms/sample")
