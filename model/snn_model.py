"""Hardware-faithful spiking neural network for a Tiny Tapeout MNIST classifier.

Every quantity in the forward pass is an integer exactly as the RTL will compute it:
  * input  : image downsampled to SxS, quantised to 2 bits (0..3), rate-coded into
             deterministic spike trains over T timesteps (done on the host)
  * weights: ternary {-1, 0, +1}, hardwired
  * neuron : LIF with arithmetic-shift leak, saturating signed membrane,
             subtractive reset:
                 v = sat(v - (v >>> LEAK) + sum_i w_i * s_i)
                 spike = v >= THETA ; if spike: v -= THETA
  * readout: spike count per output neuron over T steps, argmax (lowest index wins ties)
Surrogate gradients / straight-through estimators make it trainable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def rate_encode(levels: torch.Tensor, T: int, max_level: int = 3) -> torch.Tensor:
    """Deterministic rate code: a pixel at level q spikes q times every (max_level+1) steps,
    evenly spaced. levels: (N, P) ints -> spikes (T, N, P) float {0,1}."""
    t = torch.arange(T + 1, device=levels.device).view(-1, 1, 1)
    period = max_level + 1
    acc = torch.div(t * levels.unsqueeze(0), period, rounding_mode="floor")  # floor(t*q/period)
    return (acc[1:] > acc[:-1]).float()


def preprocess(images: torch.Tensor, size: int) -> torch.Tensor:
    """images (N,28,28) uint8 -> (N, size*size) levels in 0..3."""
    x = images.float().unsqueeze(1) / 255.0
    if size in (8, 12):
        x = x[:, :, 2:26, 2:26]  # 24x24 centre crop -> 3x3 (8x8) or 2x2 (12x12) blocks
    elif size == 10:
        x = x[:, :, 4:24, 4:24]  # 20x20 centre crop (MNIST digit box) -> 2x2 blocks
    x = F.adaptive_avg_pool2d(x, size)
    return torch.clamp(torch.floor(x * 4), 0, 3).long().flatten(1)  # 2-bit levels


class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (1 + 2.0 * x.abs()) ** 2  # fast-sigmoid surrogate (x scaled by 1/theta)


def ste_round(x):
    return x + (torch.round(x) - x).detach()


def ste_floor(x):
    return x + (torch.floor(x) - x).detach()


class TernaryLinear(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_out, n_in) * 0.4)

    def ternary(self):
        return ste_round(torch.clamp(self.weight, -1, 1))  # thresholds at +-0.5

    def forward(self, x):
        return F.linear(x, self.ternary())


class LIFLayer(nn.Module):
    def __init__(self, n_in, n_out, theta, leak_shift, vbits=8):
        super().__init__()
        self.fc = TernaryLinear(n_in, n_out)
        self.theta, self.leak_shift = theta, leak_shift
        self.vmin, self.vmax = -(2 ** (vbits - 1)), 2 ** (vbits - 1) - 1

    def forward(self, spikes):  # spikes (T, N, n_in)
        v = torch.zeros(spikes.shape[1], self.fc.weight.shape[0], device=spikes.device)
        out = []
        for s in spikes:
            if self.leak_shift > 0:
                v = v - ste_floor(v / 2 ** self.leak_shift)  # v - (v >>> k)
            v = v + self.fc(s)
            v = v + (torch.clamp(v, self.vmin, self.vmax) - v).detach()  # saturate (STE)
            z = SpikeFn.apply((v - self.theta) / self.theta)
            v = v - z.detach() * self.theta
            out.append(z)
        return torch.stack(out)


class SNN(nn.Module):
    def __init__(self, n_in, hidden, n_out=10, theta=(8, 8), leak=(4, 4), vbits=8):
        super().__init__()
        dims = [n_in] + list(hidden) + [n_out]
        self.layers = nn.ModuleList(
            LIFLayer(dims[i], dims[i + 1], theta[i], leak[i], vbits) for i in range(len(dims) - 1))

    def forward(self, spikes):
        for layer in self.layers:
            spikes = layer(spikes)
        return spikes.sum(0)  # spike counts (N, 10)
