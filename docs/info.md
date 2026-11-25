## How it works

This chip is a small **spiking neural network (SNN)** that classifies handwritten digits (MNIST) in hardware.

- **Input:** an 8×8 image with 2-bit pixel levels. The host turns it into **spike trains** over 16 timesteps: a pixel at level q spikes q times every 4 steps. Each timestep is 64 spike bits, sent as 8 bytes.
- **Network:** 64 inputs → 24 hidden → 10 output **leaky integrate-and-fire (LIF)** neurons. All 1,776 weights are ternary (−1, 0, +1) and hardwired into logic.
- **Neurons:** each timestep, `v = sat6(v − (v >>> 4) + Σ wᵢ·sᵢ)`. A neuron fires when `v ≥ θ` and then subtracts θ. θ = 8 in the hidden layer and 4 in the output layer. Membranes saturate to 6 bits.
- **Readout:** each output neuron counts its spikes over 16 timesteps. The predicted digit is the one with the most spikes, with ties going to the lower digit.

Inside:

- **Rotating neuron rings.** All neuron state lives in two shift-register rings (24 × 8-bit and 10 × 12-bit) that rotate one position every clock. Only the neuron passing the head is updated, so the whole datapath is one small adder per ring, with no per-neuron arithmetic and no enable multiplexers.
- **Weights as logic.** The trained weights are a lookup table compiled into gates, addressed by which neuron is at the head and which pixel is being processed. There's no ROM or RAM.
- **Three passes per timestep.** After the 8th byte, the chip finalises the hidden layer (fire, reset, leak), routes the hidden spikes to the output layer, then finalises the outputs and updates the running argmax. A timestep takes about 2,000 clocks, so an image takes about 3.3 ms at 10 MHz.

Accuracy is 86.83% on the 10,000-image MNIST test set. The RTL matches the Python and C++ reference models bit for bit on all 10,000 images.

## How to test

| Pin | Name | Dir | Meaning |
|---|---|---|---|
| ui[7:0] | SPIKE | in | 8 input spikes: bit i of byte b is pixel 8b+i (pixel = row·8 + col) |
| uio[0] | STROBE | in | rising edge: accept the byte on ui |
| uio[1] | CLEAR | in | rising edge: zero all neurons, counts and counters |
| uio[2] | DEBUG_MODE | in | 1: uo shows the debug word selected by uio[6:3] |
| uio[6:3] | DEBUG_SEL | in | 0–9: spike count of that digit · 10: step · 11: byte index · 15: 0xA5 |
| uo[3:0] | CLASS | out | predicted digit (final when DONE = 1) |
| uo[4] | DONE | out | all 16 timesteps received |
| uo[5] | BUSY | out | computing: wait before the next byte |
| uo[6] | STEP_START | out | the next byte begins a new timestep |

STROBE and CLEAR are synchronised inside the chip, so the host can run on its own clock. Hold each strobe level for at least 4 clocks, and keep ui steady while STROBE is high.

1. Reset the chip (`rst_n` low for at least 10 clocks with the clock running), then pulse CLEAR.
2. Encode the image: a pixel at level q ∈ {0..3} spikes at timestep t when `⌊(t+1)q/4⌋ > ⌊tq/4⌋`.
3. Send 16 timesteps × 8 bytes. For each byte: put it on ui, pulse STROBE, wait until BUSY = 0.
4. When DONE = 1, read the digit on CLASS. Extra bytes are ignored until the next CLEAR.

In Python-like pseudocode:

```python
pulse(CLEAR)
for t in range(16):
    word = sum(1 << p for p in range(64) if ((t + 1) * lv[p]) // 4 > (t * lv[p]) // 4)
    for b in range(8):
        SPIKE = (word >> (8 * b)) & 0xFF; pulse(STROBE)
        while BUSY: pass
digit = CLASS
```

**Quick check:** set DEBUG_MODE = 1 and DEBUG_SEL = 15. uo should read `0xA5`.

The test images and expected results are in `test/test_vectors.txt` (64 pixel levels, label, expected prediction, expected 10 spike counts). A MicroPython driver for the demo board is in `host/snn_demo.py` in the project repository.

## External hardware

None. Everything can be driven from the Tiny Tapeout demo board's microcontroller.
