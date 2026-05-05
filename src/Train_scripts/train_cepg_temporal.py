# === train_cepg_temporal.py ===
import os, random, numpy as np, torch, torch.nn.functional as F
from collections import Counter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv
from torch.serialization import safe_globals
import torch_scatter

# ---------- Paths ----------
DATA_PATH = "/app/codenet_cepg_ast_balanced.pt"        
#DATA_PATH = "/app/checkpoints_cepg_ast/partial_850.pt" 
MODEL_OUT = "/app/cepg_tgat.pt"
CKPT_DIR  = "/app/checkpoints_tgat"
os.makedirs(CKPT_DIR, exist_ok=True)

# ---------- Load graphs ----------
with safe_globals([Data]):
    graphs = torch.load(DATA_PATH, weights_only=False)
print(f"✅ Loaded {len(graphs)} graphs")

# sanity: consistent feature dim
in_dim = graphs[0].x.size(1)
assert all(g.x.size(1) == in_dim for g in graphs), "Inconsistent x dim"

# pick label flavor: 'combined' (default), 'status', or 'semgrep'
LABEL_TYPE = os.environ.get("CEPG_LABEL_TYPE", "combined")
def get_label_tensor(g):
    if LABEL_TYPE == "status"  and hasattr(g, "y_status"):   return g.y_status
    if LABEL_TYPE == "semgrep" and hasattr(g, "y_semgrep"):  return g.y_semgrep
    if hasattr(g, "y_combined"): return g.y_combined
    return g.y   # fallback

y_counts = Counter([int(get_label_tensor(g)) for g in graphs])
print("📊 Label counts:", y_counts)

# split
random.shuffle(graphs)
n = len(graphs)
train_split = int(0.8*n)
val_split   = int(0.9*n)
train_graphs = graphs[:train_split]
val_graphs   = graphs[train_split:val_split]
test_graphs  = graphs[val_split:]

train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_graphs, batch_size=8)
test_loader  = DataLoader(test_graphs, batch_size=8)

# ---------- Model ----------
class TGATMultiHead(torch.nn.Module):
    def __init__(self, in_channels, hidden=128, heads=4, edge_dim=1, use_semgrep_head=True, use_cls_head=True):
        super().__init__()
        self.use_semgrep_head = use_semgrep_head
        self.use_cls_head     = use_cls_head

        self.conv1 = TransformerConv(in_channels, hidden, heads=heads, edge_dim=edge_dim, dropout=0.1)
        self.conv2 = TransformerConv(hidden*heads, hidden, heads=1, edge_dim=edge_dim, dropout=0.1)

        # node-level decoders (self-supervised)
        self.reconstruct = torch.nn.Linear(hidden, in_channels)  # x̂_i
        self.next_pred   = torch.nn.Linear(hidden, in_channels)  # x̂_{i+1}

        # auxiliary semgrep regression (node-level, 1-d target)
        if use_semgrep_head:
            self.semgrep_pred = torch.nn.Linear(hidden, 1)

        # optional graph classifier head (BCE)
        if use_cls_head:
            self.readout = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index, edge_attr)
        h = F.relu(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = F.relu(h)

        x_hat   = self.reconstruct(h)
        nexthat = self.next_pred(h)
        semg    = self.semgrep_pred(h) if self.use_semgrep_head else None
        return h, x_hat, nexthat, semg

    def graph_logits(self, h, batch):
        # mean pool per graph
        g = torch_scatter.scatter_mean(h, batch, dim=0)
        return self.readout(g).squeeze(-1) if hasattr(self, "readout") else None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥️ Using device:", device)

model = TGATMultiHead(
    in_channels=in_dim,
    hidden=128, heads=4, edge_dim=1,
    use_semgrep_head=True, use_cls_head=True
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

# ---------- Helpers ----------
def get_slices(batch):
    """Retrieve feature slice mapping from batch (handles merged batches)."""
    fs = getattr(batch, "feature_slices", None)

    if fs is None:
        # fallback to known defaults
        return {"semgrep": (768+16, 768+16+1)}

    # If DataLoader merged several dicts into a list, pick the first one
    if isinstance(fs, list) and isinstance(fs[0], dict):
        fs = fs[0]

    return fs


def edge_based_next_loss(nexthat, x, edge_index):
    # use directed edges (u->v): predict x[v] from nexthat[u]
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=x.device)

    src, dst = edge_index
    # exclude root edges if you like; optional:
    # keep = (src != 0) & (dst != 0)
    # src, dst = src[keep], dst[keep]
    if src.numel() == 0:
        return torch.tensor(0.0, device=x.device)

    pred = nexthat[src]      # [E, F]
    targ = x[dst]            # [E, F]
    return F.mse_loss(pred, targ)

def bce_graph_loss(logits, y):
    if logits is None:
        return torch.tensor(0.0, device=y.device)
    yf = y.float()
    return F.binary_cross_entropy_with_logits(logits, yf)

# ---------- Train / Eval ----------
ALPHA = 0.5  # recon
BETA  = 0.3  # next
GAMMA = 0.2  # semgrep aux (node-level)
DELTA = 1.0  # graph BCE weight (supervised label)

def train_epoch(loader, epoch):
    model.train()
    total = 0.0
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        edge_attr = batch.edge_time.unsqueeze(1) if hasattr(batch, "edge_time") else torch.ones(batch.edge_index.size(1), 1, device=device)

        h, x_hat, nexthat, semg = model(batch.x, batch.edge_index, edge_attr)

        # losses
        loss_recon = F.mse_loss(x_hat, batch.x)
        loss_next  = edge_based_next_loss(nexthat, batch.x, batch.edge_index)

        # semgrep regression (use stored slice)
        fs = get_slices(batch)
        # --- Safe semgrep slice extraction ---
        semgrep_slice = fs.get("semgrep", (784, 785))

        # If it's a list of lists or tuples (like [(784, 785), (784, 785)...])
        if isinstance(semgrep_slice, (list, tuple)):
            # flatten if nested
            while isinstance(semgrep_slice, (list, tuple)) and len(semgrep_slice) > 0 and isinstance(semgrep_slice[0], (list, tuple)):
                semgrep_slice = semgrep_slice[0]
            # now try to unpack first two values
            if len(semgrep_slice) >= 2:
                s0, s1 = semgrep_slice[0], semgrep_slice[1]
            else:
                s0, s1 = 784, 785
        else:
            s0, s1 = 784, 785

        if not isinstance(s0, int) or not isinstance(s1, int):
            print("⚠️ Unexpected semgrep_slice format:", semgrep_slice)
            s0, s1 = 784, 785

        semgrep_true = batch.x[:, s0:s1]  # [N,1]
        loss_semg = torch.tensor(0.0, device=device)
        if semg is not None and semgrep_true.numel() > 0:
            # normalize target a bit to help
            loss_semg = F.mse_loss(semg, semgrep_true)

        # graph-level BCE (weakly supervised)
        # batch.y may be missing or 0-d; use our helper label
        # We need one label per graph in the batch; construct it.
        if hasattr(batch, "y"):
            y_graph = batch.y
        else:
            # If not present, we can’t do supervised loss.
            y_graph = None

        logits = model.graph_logits(h, batch.batch)
        loss_bce = torch.tensor(0.0, device=device)
        if logits is not None and y_graph is not None:
            # ensure shape [num_graphs]
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            loss_bce = bce_graph_loss(logits, y_graph)

        loss = ALPHA*loss_recon + BETA*loss_next + GAMMA*loss_semg + DELTA*loss_bce

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        total += float(loss.item())

    return total / max(1, len(loader))

@torch.no_grad()
def evaluate(loader):
    model.eval()
    # Use graph BCE logits as anomaly scores for AUC
    all_scores, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        edge_attr = batch.edge_time.unsqueeze(1) if hasattr(batch, "edge_time") else torch.ones(batch.edge_index.size(1), 1, device=device)
        h, x_hat, nexthat, semg = model(batch.x, batch.edge_index, edge_attr)
        logits = model.graph_logits(h, batch.batch)
        if logits is None or not hasattr(batch, "y"):
            continue
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels = batch.y.detach().cpu().numpy()
        all_scores.extend(probs.tolist())
        all_labels.extend(labels.tolist())

    if len(set(all_labels)) < 2:
        return None  # cannot compute AUC without both classes
    # Compute AUC
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(all_labels, all_scores)

# ---------- Main loop ----------
EPOCHS = 10
print("ℹ️ Using input dim =", in_dim, "| label_type =", LABEL_TYPE)
for e in range(1, EPOCHS+1):
    loss = train_epoch(train_loader, e)
    val_auc = evaluate(val_loader)
    if val_auc is None:
        print(f"Epoch {e} | Train Loss={loss:.4f} | Val AUC=NA (single-class in val)")
    else:
        print(f"Epoch {e} | Train Loss={loss:.4f} | Val AUC={val_auc:.4f}")
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"tgat_epoch{e}.pt"))

# final test
test_auc = evaluate(test_loader)
if test_auc is None:
    print("🎯 Test AUC=NA (single-class in test)")
else:
    print(f"🎯 Test AUC={test_auc:.4f}")

torch.save(model.state_dict(), MODEL_OUT)
print(f"✅ Saved model to {MODEL_OUT}")
