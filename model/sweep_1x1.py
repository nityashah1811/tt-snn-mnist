"""Single-layer sweep for a 1x1 tile: python sweep_1x1.py SIZE THETA LEAK VBITS"""
import json, sys, time
from pathlib import Path
import torch
import explore
torch.set_num_threads(4)
size, theta, leak, vbits = map(int, sys.argv[1:5])
orig = explore.SNN
explore.SNN = lambda n_in, hidden, theta, leak: orig(n_in, hidden, theta=theta, leak=leak, vbits=vbits)
t0 = time.time()
acc, model = explore.run(size, (), (theta,), (leak,), epochs=30)
tag = f"{size}x{size}_h0_th{theta}_lk{leak}_v{vbits}_e30"
torch.save(model.state_dict(), Path(__file__).parent / f"ckpt_{tag}.pt")
print(json.dumps({"cfg": tag, "acc": round(acc, 4), "sec": round(time.time() - t0)}), flush=True)
