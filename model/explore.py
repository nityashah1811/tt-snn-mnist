"""Sweep SNN sizes to find the best accuracy that fits in 2 Tiny Tapeout tiles.

Usage: python explore.py            (runs the built-in sweep, prints a table)
"""
import itertools
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets

from snn_model import SNN, preprocess, rate_encode

DATA = Path(__file__).resolve().parent / "data"
torch.manual_seed(0)


def load(size):
    tr = datasets.MNIST(DATA, train=True, download=True)
    te = datasets.MNIST(DATA, train=False, download=True)
    return (preprocess(tr.data, size), tr.targets), (preprocess(te.data, size), te.targets)


def run(size, hidden, theta, leak, T=16, epochs=12, bs=256, lr=5e-3, beta=0.5):
    (xtr, ytr), (xte, yte) = load(size)
    model = SNN(size * size, hidden, theta=theta, leak=leak)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * (len(xtr) // bs))
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr) - bs + 1, bs):
            idx = perm[i:i + bs]
            counts = model(rate_encode(xtr[idx], T))
            loss = F.cross_entropy(counts * beta, ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    with torch.no_grad():
        counts = torch.cat([model(rate_encode(xte[i:i + 2000], T)) for i in range(0, len(xte), 2000)])
    acc = (counts.argmax(1) == yte).float().mean().item()  # argmax -> lowest index on ties, like HW
    return acc, model


if __name__ == "__main__":
    configs = [
        # size, hidden, theta per layer, leak shift per layer
        (7, (), (8,), (4,)),
        (8, (), (8,), (4,)),
        (8, (), (16,), (3,)),
        (7, (16,), (8, 4), (4, 4)),
        (8, (16,), (8, 4), (4, 4)),
        (8, (16,), (16, 4), (3, 3)),
        (8, (24,), (8, 4), (4, 4)),
        (8, (32,), (8, 4), (4, 4)),
    ]
    for size, hidden, theta, leak in configs:
        t0 = time.time()
        acc, model = run(size, hidden, theta, leak)
        tag = f"{size}x{size}_h{'-'.join(map(str, hidden)) or 0}_th{'-'.join(map(str, theta))}_lk{'-'.join(map(str, leak))}"
        torch.save(model.state_dict(), Path(__file__).parent / f"ckpt_{tag}.pt")
        print(json.dumps({"cfg": tag, "acc": round(acc, 4), "sec": round(time.time() - t0)}), flush=True)
