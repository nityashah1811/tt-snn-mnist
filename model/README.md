# Model: training, integer reference and golden vectors

| File | What |
|---|---|
| `snn_model.py` | PyTorch SNN whose forward pass is exactly the chip's integer maths (ternary weights, `v >>> 4` leak, 6-bit saturation, subtractive reset), trained with surrogate gradients and straight-through estimators |
| `explore.py` | size/accuracy sweep for two-layer networks. Results in `sweep1.log` (12 epochs) and `sweep2.log` (30 epochs) |
| `sweep_1x1.py` | single-layer sweep for a 1x1 tile (`python sweep_1x1.py SIZE THETA LEAK VBITS`). Results in `sweep_1x1.log` |
| `ckpt_10x10_h0_th8_lk4_v8_e30.pt` | the trained checkpoint on the chip (100-10, 8-bit membranes, θ = 8, 87.85%) |
| `export.py` | runs a pure-NumPy integer reference, asserts it matches PyTorch on all 10,000 test images, then writes `weights_ternary.npz`, `snn_weights.h` and `test_vectors.txt` |
| `weights_ternary.npz` | the exported ternary weights: `arr_0` is W (10×100) |
| `test_vectors.txt` | all 10,000 test images: 100 levels, label, golden prediction, golden 10 spike counts |
| `gen_weights.py` | generates `../src/snn_weights.v` from `weights_ternary.npz` |
| `cpp/` | synthesizable C++ golden model with testbench and Vitis HLS script (`run_hls.tcl`) |

MNIST is downloaded to `model/data/` on first use (not committed).

```bash
pip install torch torchvision numpy
python sweep_1x1.py 10 8 4 8
python export.py --ckpt ckpt_10x10_h0_th8_lk4_v8_e30.pt --size 10 --hidden 0 --theta 8 --leak 4 --vbits 8 --out .
python gen_weights.py
```

C++ check without Vitis:

```bash
cd cpp && g++ -std=c++11 -O2 snn.cpp tb.cpp -o tb && ./tb ../test_vectors.txt
```

Expected output: `vectors: 10000  bit-exact mismatches: 0  accuracy vs labels: 87.85%`, then `PASS`.
