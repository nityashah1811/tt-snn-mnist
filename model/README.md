# Model: training, integer reference and golden vectors

| File | What |
|---|---|
| `snn_model.py` | PyTorch SNN whose forward pass is exactly the chip's integer maths (ternary weights, `v >>> 4` leak, 6-bit saturation, subtractive reset), trained with surrogate gradients and straight-through estimators |
| `explore.py` | size/accuracy sweep. Results in `sweep1.log` (12 epochs) and `sweep2.log` (30 epochs) |
| `ckpt_8x8_h24_v6_e30.pt` | the trained checkpoint on the chip (64-24-10, 6-bit membranes, 86.83%) |
| `export.py` | runs a pure-NumPy integer reference, asserts it matches PyTorch on all 10,000 test images, then writes `weights_ternary.npz`, `snn_weights.h` and `test_vectors.txt` |
| `weights_ternary.npz` | the exported ternary weights: `arr_0` is W1 (24×64), `arr_1` is W2 (10×24) |
| `test_vectors.txt` | all 10,000 test images: 64 levels, label, golden prediction, golden 10 spike counts |
| `gen_weights.py` | generates `../src/snn_weights.v` from `weights_ternary.npz` |
| `cpp/` | synthesizable C++ golden model with testbench and Vitis HLS script (`run_hls.tcl`) |

MNIST is downloaded to `model/data/` on first use (not committed).

```bash
pip install torch torchvision numpy
python explore.py
python export.py --ckpt ckpt_8x8_h24_v6_e30.pt --hidden 24 --theta 8 4 --leak 4 4 --vbits 6 --out .
python gen_weights.py
```

C++ check without Vitis:

```bash
cd cpp && g++ -std=c++11 -O2 snn.cpp tb.cpp -o tb && ./tb ../test_vectors.txt
```

Expected output: `vectors: 10000  bit-exact mismatches: 0  accuracy vs labels: 86.83%`, then `PASS`.
