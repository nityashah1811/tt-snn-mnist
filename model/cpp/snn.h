// Spiking neural network MNIST classifier -- synthesizable C++ for HLS.
//
// Semantics (bit-exact with export.py's integer reference), per timestep, per layer:
//     v     = v - (v >>> LEAK)                       arithmetic-shift leak
//     v     = sat(v + popcnt(s & POS) - popcnt(s & NEG))   full sum first, saturate ONCE
//     spike = (v >= THETA);  if (spike) v -= THETA   subtractive reset
// Layer 2 sees layer 1's spikes from the SAME timestep. Output = per-class spike count
// over SNN_T steps; prediction = argmax, lowest index wins ties.
//
// Plain <stdint.h> types only: no floats, no dynamic memory, all loops have constant bounds.
#pragma once
#include <stdint.h>
#include "snn_weights.h"

#if SNN_H > 0
typedef uint32_t hidden_t;  // hidden-layer spike vector (SNN_H <= 32)
#endif

struct SnnState {
#if SNN_H > 0
    int8_t v1[SNN_H];      // SNN_VBITS-bit signed membranes
#endif
    int8_t v2[SNN_OUT];
    uint8_t count[SNN_OUT];  // 0..SNN_T  (5 bits for T=16)
};

// --- Stateful step API (natural fit for a pin-limited Tiny Tapeout wrapper) ---
void snn_reset(SnnState &st);
void snn_timestep(SnnState &st, uint64_t in_spikes);  // bit p = pixel p spiked this step
uint8_t snn_argmax(const SnnState &st);

// --- Whole-inference top function: SNN_T spike words in, class (0-9) out ---
uint8_t snn_infer(const uint64_t spikes[SNN_T], uint8_t counts_out[SNN_OUT]);

// --- HLS top for Tiny Tapeout (Vitis HLS set_top snn_step_top) ---
// Call once per timestep with that step's 64 spike bits; neuron state lives inside the block.
// clear=1 on the first step of an image zeroes all state before integrating that step.
// counts_out = running spike counts; return = running argmax (final after SNN_T calls).
uint8_t snn_step_top(uint64_t spike_word, bool clear, uint8_t counts_out[SNN_OUT]);

// Host-side helper (NOT for synthesis): deterministic rate code of 2-bit pixel levels.
// Pixel at level q (0..3) spikes at step t iff floor((t+1)q/4) > floor(tq/4).
void snn_rate_encode(const uint8_t levels[SNN_IN], uint64_t spikes[SNN_T]);
