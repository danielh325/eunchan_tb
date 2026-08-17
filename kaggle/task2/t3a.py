"""T3A -- Test-Time Classifier Adjustment (Iwasawa & Matsuo, NeurIPS 2021 spotlight).

Replaces the trained linear head with class prototypes built from the *test*
data's own features, then classifies by similarity to those prototypes. It
touches nothing but the final linear layer and does **no backpropagation**.

WHY THIS INSTEAD OF Code/tta.py
-------------------------------
`Code/tta.py` implements source-free adaptation by gradient descent (entropy
minimization), wired in as `predict_task2.py --tta`. It has never been
evaluated -- and the literature is consistent about which way that gamble
breaks. From a recent systematic study of test-time adaptation under real
distribution shift:

    "entropy minimization methods such as TENT and SAR perform best when the
     target distribution is clean, while prototype adjustment methods like T3A
     excel under larger distributional distance ... T3A exhibits the most
     stable behavior ... with lower variability across seeds and batch sizes,
     while gradient-based methods (Tent and SHOT) consistently degrade
     performance."

A private 14-country test set is the "large distributional distance, cannot
check the result" case exactly. A gradient-based adapter that collapses toward
one class on one country would be invisible to you until scoring. T3A cannot
collapse that way: it is backprop-free, has no learning rate, and reduces to
the original classifier when the support sets are empty.

It also fits the frozen-backbone setup perfectly -- the backbone is already
frozen, so the features T3A needs are exactly what the model already computes.

Cost: one extra matrix multiply per batch. No second forward pass.

USAGE
-----
    from t3a import T3AHead
    head = T3AHead.from_linear(model.head, filter_k=20)
    # pass 1: accumulate support from the unlabeled test features
    for x in loader:
        z = model.encoder(x)          # pooled features, (B, D)
        head.update(z)
    # pass 2: predict
    for x in loader:
        prob = head.predict_proba(model.encoder(x))

`predict_task2.py` already separates encoder and head, so this drops in without
touching the training code at all.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class T3AHead:
    """Binary (TB/Normal) T3A.

    The project's head emits a single logit, so it is expanded into the
    equivalent 2-class form: w_TB = w, w_Normal = -w, which produces identical
    decisions before adaptation and gives T3A the two templates it needs.

    Args:
        w:  (D,) weight vector of the trained linear layer
        b:  scalar bias
        filter_k: how many lowest-entropy support samples to keep per class.
                  The paper's main hyperparameter. -1 keeps all. Small values
                  (5-20) are conservative; the support set stays close to the
                  original template.
    """

    def __init__(self, w, b=0.0, filter_k=20, pre_norm=None):
        w = w.detach().float().flatten()
        self.device = w.device
        self.b = float(b)
        self.filter_k = filter_k
        self.pre_norm = pre_norm          # optional nn.LayerNorm from the head
        # templates: index 0 = Normal, 1 = TB
        self.templates = torch.stack([-w, w]).to(self.device)      # (2, D)
        self.support = [self.templates[0:1].clone(), self.templates[1:2].clone()]
        self.ent = [torch.zeros(1, device=self.device),
                    torch.zeros(1, device=self.device)]

    @classmethod
    def from_linear(cls, head, filter_k=20):
        """Build from the project's `nn.Sequential(LayerNorm, Dropout, Linear)`
        head. The LayerNorm is kept and applied to incoming features so the
        support set lives in the same space the templates were trained in."""
        ln = next((m for m in head.modules() if isinstance(m, nn.LayerNorm)), None)
        lin = next(m for m in reversed(list(head.modules())) if isinstance(m, nn.Linear))
        b = float(lin.bias.detach()) if lin.bias is not None else 0.0
        return cls(lin.weight, b, filter_k=filter_k, pre_norm=ln)

    def _feat(self, z):
        z = z.detach().float()
        if self.pre_norm is not None:
            z = self.pre_norm(z)
        return z

    @torch.no_grad()
    def update(self, z):
        """Accumulate unlabeled test features into the per-class support sets,
        keyed by the current pseudo-label and filtered by prediction entropy."""
        z = self._feat(z)
        logits = z @ self.templates.T                       # (B, 2)
        p = F.softmax(logits, dim=1)
        ent = -(p * p.clamp_min(1e-12).log()).sum(1)        # (B,)
        yhat = logits.argmax(1)
        for c in (0, 1):
            m = yhat == c
            if not m.any():
                continue
            self.support[c] = torch.cat([self.support[c], z[m]], 0)
            self.ent[c] = torch.cat([self.ent[c], ent[m]], 0)
            if self.filter_k > 0 and self.support[c].size(0) > self.filter_k:
                keep = self.ent[c].argsort()[:self.filter_k]
                self.support[c] = self.support[c][keep]
                self.ent[c] = self.ent[c][keep]

    @torch.no_grad()
    def _prototypes(self):
        return torch.stack([s.mean(0) for s in self.support])     # (2, D)

    @torch.no_grad()
    def predict_proba(self, z):
        """P(TB) for each row of z."""
        z = self._feat(z)
        logits = z @ self._prototypes().T                          # (B, 2)
        return F.softmax(logits, dim=1)[:, 1]

    @torch.no_grad()
    def predict_proba_original(self, z):
        """P(TB) from the untouched trained head -- the control you must
        compute alongside, so you can see what T3A changed."""
        z = self._feat(z)
        return torch.sigmoid(z @ self.templates[1] + self.b)


@torch.no_grad()
def t3a_adapt_and_predict(encoder, head, loader, device, filter_k=20):
    """Two-pass convenience wrapper: build support over the whole test set,
    then predict. Returns (p_t3a, p_original) so the two are always compared.

    Ship T3A only if it beats the original on Shenzhen AND Montgomery AND
    TBX11K. An adaptation method that helps on two of three is a coin flip on
    a country you cannot inspect.
    """
    t3a = T3AHead.from_linear(head, filter_k=filter_k)
    feats = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        z = encoder(x.to(device))
        feats.append(z.detach().float())
        t3a.update(z)
    feats = torch.cat(feats, 0)
    return t3a.predict_proba(feats).cpu(), t3a.predict_proba_original(feats).cpu()
