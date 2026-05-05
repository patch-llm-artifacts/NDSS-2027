# tgat_classifier.py
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TGATClassifier(nn.Module):
    """
    Lightweight TGAT proxy classifier.
    Uses ONLY the trained readout (Dropout + Linear).
    Works without graph, without TransformerConv,
    and without edge_index / edge_attr.
    """

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h):
        """
        h : Tensor [hidden_dim]
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)
        return self.classifier(h).squeeze(-1)


def load_tgat_classifier(ckpt="/app/tgat_juliet_v3_best.pt"):
    """
    Extracts ONLY readout.1.weight and readout.1.bias from TGAT checkpoint.
    """
    state = torch.load(ckpt, map_location=device)

    # create classifier
    clf = TGATClassifier(hidden_dim=256).to(device)

    # extract readout weights
    cleaned = {
        "classifier.1.weight": state["readout.1.weight"],
        "classifier.1.bias": state["readout.1.bias"],
    }

    clf.load_state_dict(cleaned, strict=True)
    clf.eval()

    print("✅ Loaded TGAT readout classifier (graph-free)")
    return clf
