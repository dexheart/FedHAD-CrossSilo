# -*- coding: utf-8 -*-
"""
Global-gradient and class-sensitive diagnostics (reviewer R2, Recommendation94).

Definitions, all at the global model w^t of round t (before local training):
  g_k      gradient of the local training objective F_k (cross-entropy, no proximal
           term) over client k's whole training split, model in eval mode (BatchNorm
           running statistics, no dropout), so the gradient is deterministic;
  g        sum_k p_k g_k with p_k = n_k / N: the gradient of the global training
           objective over the union of the clients' training splits;
  g_-k     (N g - n_k g_k) / (N - n_k): the global gradient WITHOUT client k, so a
           client is never compared with itself;
  u_k      w_k^{t+1} - w^t, the update the client actually returns (trainable
           parameters only); its descent direction is compared with -g_-k;
  epoch e  w_{k,e} - w_{k,e-1}: the movement of local epoch e, compared with -g_-k
           ("marginal alignment") together with the first-order change it implies in
           the loss of the rest of the federation, <w_{k,e} - w_{k,e-1}, g_-k>
           (negative = that epoch decreases the others' loss to first order).

Only trainable parameters enter (BatchNorm buffers excluded), flattened in
net.parameters() order on both server and client. The functions below are pure torch.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def flat_params(net):
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()])


def flat_grad(net):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1)
                      for p in net.parameters()])


def cosine(a, b):
    na, nb = float(a.norm()), float(b.norm())
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(torch.dot(a, b) / (na * nb))


def full_gradient(net, loader, device):
    """Mean cross-entropy gradient over every sample of `loader` (eval mode)."""
    net.eval()
    total, acc = 0, None
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        net.zero_grad(set_to_none=True)
        loss = F.cross_entropy(net(images), labels, reduction="sum")
        loss.backward()
        g = flat_grad(net)
        acc = g if acc is None else acc + g
        total += int(labels.shape[0])
    net.zero_grad(set_to_none=True)
    if total == 0:
        return None, 0
    return acc / total, total


def loo_gradients(local_grads, n_samples):
    """g and g_-k from per-client mean gradients (None for clients without data)."""
    N = float(sum(n for g, n in zip(local_grads, n_samples) if g is not None))
    g = sum((n / N) * gk for gk, n in zip(local_grads, n_samples) if gk is not None)
    loo = []
    for gk, n in zip(local_grads, n_samples):
        loo.append(None if gk is None or N - n <= 0 else (N * g - n * gk) / (N - n))
    return g, loo


def confusion(net, loader, n_classes, device):
    cm = torch.zeros((n_classes, n_classes), dtype=torch.long)
    net.eval()
    with torch.no_grad():
        for images, labels in loader:
            pred = net(images.to(device)).argmax(1).cpu()
            for t, p in zip(labels.view(-1), pred.view(-1)):
                cm[int(t), int(p)] += 1
    return cm.numpy()


def class_metrics(cm):
    """Per-class recall, worst-class recall, macro-F1 and per-class recall SD (pp)."""
    cm = np.asarray(cm, dtype=float)
    support = cm.sum(1)
    tp = np.diag(cm)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    pred = cm.sum(0)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {"per_class_recall": [float(x) for x in recall],
            "worst_class_recall": float(recall.min()),
            "macro_f1": float(f1.mean()),
            "recall_sd": float(recall.std()),
            "accuracy": float(tp.sum() / cm.sum()) if cm.sum() else float("nan")}


class EpochTracker:
    """Client-side: alignment of each local epoch with -g_-k (and -g)."""

    def __init__(self, net, g_loo, g_all):
        self.net = net
        self.g_loo = g_loo
        self.g_all = g_all
        self.w0 = flat_params(net).clone()
        self.prev = self.w0.clone()
        self.rows = []

    def __call__(self, epoch):
        w = flat_params(self.net)
        step, cum = w - self.prev, w - self.w0
        r = {"epoch": int(epoch), "step_norm": float(step.norm())}
        if self.g_loo is not None:
            r.update({"marg_cos_loo": cosine(step, -self.g_loo),
                      "cum_cos_loo": cosine(cum, -self.g_loo),
                      "marg_lin_loo": float(torch.dot(step, self.g_loo))})
        if self.g_all is not None:
            r.update({"marg_cos_all": cosine(step, -self.g_all),
                      "marg_lin_all": float(torch.dot(step, self.g_all))})
        self.rows.append(r)
        self.prev = w.clone()

    def as_metrics(self, max_epochs=5):
        """Flower metrics must be scalars: one key per (quantity, epoch)."""
        out = {}
        for r in self.rows:
            e = r["epoch"]
            if e > max_epochs:
                continue
            for k, v in r.items():
                if k != "epoch" and v is not None and not (isinstance(v, float) and math.isnan(v)):
                    out[f"gg_{k}_e{e}"] = float(v)
        return out


def write_rows(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
