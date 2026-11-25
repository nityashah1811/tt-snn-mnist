![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# SNN MNIST Digit Classifier

## Overview

A **spiking neural network** in silicon for [Tiny Tapeout](https://tinytapeout.com) (sky130), in a single 1x1 tile. It classifies handwritten MNIST digits with 100 input spikes, 10 leaky integrate-and-fire neurons, and 1,000 hardwired ternary weights.

**Author:** Nitya Shah

| | |
|---|---|
| Size | 1x1 tile |
| Clock | 10 MHz nominal (timing signed off at 50 MHz) |
| Accuracy | 87.85% on the 10,000-image MNIST test set |
| Verification | RTL bit-exact with the reference model on all 10,000 test images |

How it works and how to use it: [docs/info.md](docs/info.md)

---

## Contents

- [At a glance](#at-a-glance)
- [The network](#the-network)
- [Architecture](#architecture)
- [How it works](#how-it-works)
- [Pinout](#pinout)
- [Talking to the chip](#talking-to-the-chip)
- [Verification](#verification)
- [Physical design](#physical-design)
- [From training to silicon](#from-training-to-silicon)
- [Build it yourself](#build-it-yourself)
- [Repository map](#repository-map)

## At a glance

| | |
|---|---|
| **Function** | classify a 10×10 handwritten digit as 0–9 |
| **Network** | 100 inputs → 10 LIF neurons, spike-count readout |
| **Weights** | 1,000 ternary weights (−1, 0, +1), 712 non-zero, hardwired as logic |
| **Input** | 10×10 image at 2-bit levels, rate-coded into 16 timesteps × 100 spikes, sent as 208 bytes |
| **Neurons** | 8-bit saturating membranes, shift-based leak `v >>> 4`, subtractive reset |
| **Compute time** | about 1,200 clocks per timestep, so one image takes about 2 ms at 10 MHz |
| **Arithmetic** | one 9-bit adder for all 10 neurons, 0 multipliers |
| **Storage** | 191 flip-flops, no RAM |
| **Accuracy** | 87.85% on MNIST test (10,000 images), digits 0 and 1 above 96% |
| **Process** | SkyWater sky130A, `sky130_fd_sc_hd` standard cells |
| **Language** | Verilog, about 140 lines of RTL (200 with comments) plus a generated weight table |
| **License** | Apache-2.0 |

## The network

![Spikes flowing through the network](docs/img/spikes.svg)

A spiking network computes with **events over time** instead of numbers in one shot. Each pixel becomes a train of spikes: a brighter pixel spikes more often. Every neuron keeps a **membrane voltage** that rises with weighted input spikes, slowly **leaks** back toward zero, and **fires** a spike of its own when it crosses a threshold. The digit is whichever neuron fires most often over 16 timesteps.

The figure above is real data: test image 0 (a 7) run through the exact integer model the chip implements ([`model/make_figures.py`](model/make_figures.py)). The input pattern repeats every 4 steps because of the rate code. Neuron 7 fires in 14 of 16 timesteps. The next closest, neuron 9, fires 3 times.

| | Size | Threshold θ | Leak | Membrane |
|---|---:|---:|---|---|
| Input | 100 (10×10 pixels) | | rate code, period 4 | |
| Neurons | 10 | 8 | `v −= v >>> 4` | 8-bit, saturating, plus a 5-bit spike counter |

The input is the central 20×20 pixels of the 28×28 MNIST image (where MNIST centres the digit), averaged over 2×2 blocks and quantised to 4 levels. The host does this preprocessing.

### Accuracy by digit

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Correct | 96.9% | 97.4% | 83.5% | 89.1% | 92.2% | 76.9% | 91.1% | 88.0% | 83.5% | 77.9% |

The most common mistake by far is 9 → 4 (119 of the 1,215 errors), followed by 5 → 3, 2 → 8 and 5 → 8.

### Why this network?

The chip had to fit **one tile**. In this architecture, neuron state costs flip-flops, but extra inputs only cost weight logic, because pixels stream in from the host and are never stored. So the best use of a small tile is **no hidden layer and a sharper input**. A sweep (in [`model/`](model/)) bears this out:

| Network | Weights | Neuron state bits | Test accuracy | Fits |
|---|---:|---:|---:|---|
| 8×8 → 10 | 640 | 140 | 79.3% | 1x1 |
| **10×10 → 10** | **1,000** | **140** | **87.85%** | **1x1** |
| 12×12 → 10 | 1,440 | 150 | 88.0% | not tried: +44% weight logic for +0.1% |
| 14×14 → 10 | 1,960 | 150 | 88.0% | not tried: about 2× the weight logic |
| 8×8 → 24 → 10 | 1,776 | 312 | 86.8% | 1x2 only |

10×10 gets almost all of the benefit of a sharper image. The earlier two-layer 1x2 version of this chip is kept at tag [`v1x2-h24`](../../tree/v1x2-h24). It hardened cleanly, but the single layer at 10×10 beats it in half the area.

## Architecture

![Block diagram](docs/img/architecture.svg)

Two files of RTL:

| File | Module | What it does |
|---|---|---|
| [`src/project.v`](src/project.v) | `tt_um_nityashah1811_snn_mnist` | pins, synchronisers, FSM, neuron ring, LIF datapath, argmax, debug readout |
| [`src/snn_weights.v`](src/snn_weights.v) | `snn_w` | the trained ternary weights as a lookup table (generated) |

### Where the 191 flip-flops go

Every flop is accounted for, and the total matches the synthesised netlist exactly.

| Block | Flops | Notes |
|---|---:|---|
| Neuron ring | 140 | 10 neurons × (9-bit accumulator + 5-bit spike count) |
| Input byte | 8 | the latched spike byte |
| Synchronisers | 6 | 2 strobes × 3 stages |
| Control | 37 | FSM state, ring pointer, pixel/byte/step counters, argmax, prediction, debug |
| **Total** | **191** | |

## How it works

### 1. Rotating neuron ring: one adder for 10 neurons

All neuron state lives in a **shift-register ring** that rotates one position **every clock**, whether or not anything is happening. Each clock, the neuron at the head is read, optionally updated, and written back at the tail. A counter `op` tracks which neuron is at the head.

Rotating every clock is deliberate. A ring that only moves when told to needs an enable multiplexer on every one of its 140 flops. A free-running ring needs none, so all the arithmetic sits in one small datapath at the head. The FSM just waits until neuron 0 comes round before starting a pass.

### 2. Integrate: one pixel per rotation

When a spike byte arrives, the ring makes **8 full rotations**, one per pixel in the byte. On rotation `pix`, every neuron passing the head adds its weight for that pixel if the pixel spiked:

```
acc += din[pix] ? W[neuron][8*byte + pix] : 0          (W is −1, 0 or +1)
```

The weight comes from a 1,000-entry table **compiled into logic**, addressed by `{op, byte, pix}`. There's no ROM to load and no memory to get wrong. Synthesis folds the constants into gates.

The accumulator is 9 bits wide, so the membrane (at most ±128) plus a whole timestep of input (at most ±100) can never overflow. Saturation happens once, at the end of the timestep, exactly as in the trained model.

### 3. Fire: finalise once per timestep

After the 13th byte, a single rotation (**FIN**) finalises all 10 neurons:

```
v = sat8(acc)            clamp to [−128, 127]
s = (v ≥ 8)              fire?
count += s               5-bit spike counter
v = v − 8·s              subtractive reset
v = v − (v >>> 4)        leak, applied now for the next timestep
```

The same pass updates a running argmax (ties go to the lower digit), so `CLASS` is valid as soon as the last timestep finishes.

Applying the leak right after the reset gives exactly the same result as applying it at the start of the next step, because nothing touches the neuron in between. This saves a pass.

### 4. Neurons, timestep by timestep

Two neurons on test image 0, from the integer model the chip matches:

| Step | | v at start | after leak | input Σw·s | v after sat8 | fires? | v after reset |
|---:|---|---:|---:|---:|---:|:---:|---:|
| 1 | neuron 7 | 0 | 0 | +11 | 11 | **yes** | 3 |
| 2 | | 3 | 3 | +6 | 9 | **yes** | 1 |
| 3 | | 1 | 1 | +13 | 14 | **yes** | 6 |
| 4 | | 6 | 6 | 0 | 6 | | 6 |
| 1 | neuron 4 | 0 | 0 | −3 | −3 | | −3 |
| 2 | | −3 | **−2** | −4 | −6 | | −6 |
| 3 | | −6 | −5 | −2 | −7 | | −7 |
| 4 | | −7 | −6 | 0 | −6 | | −6 |

Neuron 7 (the right answer) fires almost every step. Neuron 4 is pushed negative, and the leak pulls it back toward zero: `−3 >>> 4 = −1`, so −3 leaks to −2. The chip, the C++ model and the PyTorch training model all agree on this rounding.

### 5. Clock budget

| Phase | Clocks | Notes |
|---|---:|---|
| Load one byte | ≤ 9 + 80 | align to neuron 0, then 8 rotations of 10 |
| FIN | ≤ 9 + 10 | one rotation |
| **One timestep** | **about 1,200** | 13 bytes plus the finalise pass |
| **One image** | **about 19,000** | about 2 ms at 10 MHz |

With a microcontroller bit-banging the pins, the host is the bottleneck, not the chip.

## Pinout

| Pin | Name | Dir | Meaning |
|---|---|:---:|---|
| `ui[7:0]` | `SPIKE[7:0]` | in | 8 input spikes: bit i of byte b is pixel 8b+i (pixel = row·10 + col; bits 4–7 of byte 12 are ignored) |
| `uio[0]` | `STROBE` | in | rising edge: accept the byte on `ui` |
| `uio[1]` | `CLEAR` | in | rising edge: zero all neurons, counts and counters |
| `uio[2]` | `DEBUG_MODE` | in | 1: `uo` shows the debug word selected by `uio[6:3]` |
| `uio[6:3]` | `DEBUG_SEL` | in | 0–9: spike count of that digit · 10: step counter · 11: byte index · 15: `0xA5` |
| `uio[7]` | | | unused |
| `uo[3:0]` | `CLASS` | out | predicted digit (running argmax, final when `DONE`) |
| `uo[4]` | `DONE` | out | all 16 timesteps received |
| `uo[5]` | `BUSY` | out | computing: wait before the next byte |
| `uo[6]` | `STEP_START` | out | the next byte begins a new timestep |
| `uo[7]` | | out | 0 |

All `uio` pins are inputs (`uio_oe = 0`). `STROBE` and `CLEAR` pass through 2-flop synchronisers with edge detection, so the host can be a slow microcontroller on its own clock. Hold each strobe level for at least 4 clocks, and keep `ui` steady while `STROBE` is high.

## Talking to the chip

```mermaid
sequenceDiagram
    participant H as Host (RP2040)
    participant C as SNN chip
    H->>C: reset (rst_n low ≥ 10 clocks)
    H->>C: CLEAR pulse
    loop 16 timesteps
        loop 13 bytes
            H->>C: SPIKE byte + STROBE pulse
            C-->>H: BUSY (about 90 clocks)
        end
        Note over C: after byte 13: FIN (about 20 clocks)
    end
    C-->>H: DONE = 1, CLASS = predicted digit
    opt debug
        H->>C: DEBUG_MODE = 1, DEBUG_SEL = k
        C-->>H: spike count of digit k
    end
```

1. **Reset** (`rst_n` low for at least 10 clocks, clock running), then pulse **CLEAR**.
2. **Prepare** the image: take the central 20×20 pixels of the 28×28 digit, average each 2×2 block, and quantise to q = min(3, ⌊4·mean/255⌋).
3. **Encode**: a pixel at level q spikes at timestep t when `⌊(t+1)q/4⌋ > ⌊tq/4⌋`.
4. **Send** 16 timesteps × 13 bytes. For each byte: put it on `ui`, raise `STROBE` for at least 4 clocks, lower it, then wait for `BUSY = 0`.
5. **Read** `CLASS` once `DONE = 1`. Extra bytes are ignored until the next `CLEAR`.

```python
def encode(levels):                       # 100 levels, 0..3
    return [sum(1 << p for p in range(100)
                if ((t + 1) * levels[p]) // 4 > (t * levels[p]) // 4)
            for t in range(16)]

pulse(CLEAR)
for word in encode(levels):
    for b in range(13):
        SPIKE = (word >> (8 * b)) & 0xFF; pulse(STROBE)
        while BUSY: pass
digit = CLASS                             # DONE is now 1
```

A MicroPython driver for the Tiny Tapeout demo board is in [`host/snn_demo.py`](host/snn_demo.py). It hasn't been run on silicon yet.

## Verification

The chip is checked against the same integer model at every level, from PyTorch down to the gate-level netlist. Every check compares the predicted digit **and all 10 spike counts**.

```
PyTorch model (training) ──bit-exact──► NumPy integer model ──► test_vectors.txt (10,000 images)
                                                                  │
            C++ golden model (model/cpp)                      ◄────┤
            Verilog RTL, pin-level, through the real protocol ◄────┤
            Yosys gate-level netlist, same testbench          ◄────┤
            sky130 gate-level netlist (cocotb, GitHub CI)     ◄────┘
```

| Check | Result |
|---|---|
| PyTorch model vs NumPy integer model | identical spike counts on all 10,000 test images |
| C++ model ([`model/cpp/`](model/cpp/)) | 10,000 / 10,000 bit-exact |
| **RTL**, pin-level testbench ([`sim/`](sim/)) | **10,000 / 10,000 bit-exact**, 87.85% accuracy |
| Yosys gate netlist, same testbench | 2,000 / 2,000 bit-exact |
| Protocol edge cases (RTL and gates) | clear mid-image, clear while computing, reset mid-computation, strobe/clear held across reset, bytes after done, random handshake timing |
| cocotb in CI ([`test/test.py`](test/test.py)) | RTL and the routed sky130 gate-level netlist |
| Structural | no latches, no combinational loops, no Yosys warnings |

The local pin-level testbench uses [Yosys CXXRTL](https://yosyshq.readthedocs.io/projects/yosys/en/latest/cmd/write_cxxrtl.html), which compiles the design into a cycle-accurate C++ simulator. About 230 million clock cycles were simulated in total.

## Physical design

*Layout render, utilisation, timing across corners and signoff results will be added here from the GitHub `gds` action.*

| | |
|---|---|
| Pre-layout cell area | about 7,760 µm² (Yosys, sky130_fd_sc_hd cell areas; 191 flip-flops = 3,820 µm²) |
| 1x1 core | about 16,500 µm² |
| Clock | signed off at 50 MHz (`CLOCK_PERIOD = 20` ns), operated at 10 MHz |

For reference, the 8×8 version of this same design hardened cleanly in 1x1 at 74.6% utilisation, with 0 setup/hold violations, 0 DRC errors and the gate-level tests passing.

## From training to silicon

| Stage | Where | What |
|---|---|---|
| 1. Train | [`model/snn_model.py`](model/snn_model.py), [`model/sweep_1x1.py`](model/sweep_1x1.py) | PyTorch SNN whose forward pass **is** the integer hardware maths (ternary weights, shift leak, saturation), trained with surrogate gradients |
| 2. Export | [`model/export.py`](model/export.py) | checks the integer model against PyTorch, writes weights and `test_vectors.txt` |
| 3. Golden C++ | [`model/cpp/`](model/cpp/) | synthesizable C++ reference (Vitis HLS script included) |
| 4. RTL | [`src/`](src/), [`model/gen_weights.py`](model/gen_weights.py) | hand-written Verilog; weights generated into `snn_weights.v` |
| 5. Verify | [`sim/`](sim/), [`test/`](test/) | pin-level simulation on RTL and gates, cocotb in CI |
| 6. Harden | GitHub `gds` action | LibreLane on sky130A |

An HLS-generated RTL was tried for the first version and **not** used: it needed about 6,800 flip-flops and stored the weights in `$readmemh` ROMs, which an ASIC can't load. The C++ remains the golden model, and the hand-written RTL matches it bit for bit.

## Build it yourself

**cocotb tests** (RTL), the same tests CI runs:

```bash
cd test && pip install -r requirements.txt && make -B
```

**Full pin-level regression** (all 10,000 images on RTL, 2,000 on gates). Needs no simulator install:

```bash
pip install yowasp-yosys ziglang
bash sim/run_sim.sh
```

**Retrain and re-export:**

```bash
cd model
python sweep_1x1.py 10 8 4 8          # 10x10 input, theta 8, leak 4, 8-bit membranes
python export.py --ckpt ckpt_10x10_h0_th8_lk4_v8_e30.pt --size 10 --hidden 0 --theta 8 --leak 4 --vbits 8 --out .
mv snn_weights.h cpp/
python gen_weights.py                 # regenerates ../src/snn_weights.v
python make_figures.py                # regenerates ../docs/img/spikes.svg
```

**Harden to GDS:** push to GitHub. The `gds` action builds the layout, runs the precheck and the gate-level tests, and publishes the viewer.

## Repository map

```
src/        RTL: project.v (top), snn_weights.v (generated weights), config.json (flow settings)
test/       cocotb tests (RTL + gate level) and golden test vectors
sim/        CXXRTL pin-level testbench, full 10,000-image regression
model/      PyTorch training, integer reference, export, weight generator, checkpoint, figures
model/cpp/  synthesizable C++ golden model and Vitis HLS script
host/       MicroPython driver for the Tiny Tapeout demo board
docs/       info.md (the Tiny Tapeout datasheet page) and README figures
gds/        layout render, metrics and signoff reports from the gds action
info.yaml   Tiny Tapeout project metadata and pinout
```

Released under the [Apache-2.0 license](LICENSE).
