#include "snn.h"

// Bit-loop popcount: portable across HLS tools (unrolls to an adder tree).
static inline int popcount64(uint64_t x) {
    int c = 0;
    for (int i = 0; i < 64; i++) c += (int)((x >> i) & 1u);
    return c;
}

#if SNN_H > 0
static inline int popcount32(uint32_t x) {
    int c = 0;
    for (int i = 0; i < 32; i++) c += (int)((x >> i) & 1u);
    return c;
}
#endif

// Ternary dot product of a multi-word input spike vector with one neuron's weights.
static inline int dot_in(const spikes_t &s, const uint64_t pos[SNN_IW], const uint64_t neg[SNN_IW]) {
    int acc = 0;
    for (int k = 0; k < SNN_IW; k++) acc += popcount64(s.w[k] & pos[k]) - popcount64(s.w[k] & neg[k]);
    return acc;
}

// floor(v / 2^k) without relying on implementation-defined signed >>.
static inline int asr(int v, int k) { return v >= 0 ? (v >> k) : ~((~v) >> k); }

static inline int sat(int v) { return v < SNN_VMIN ? SNN_VMIN : (v > SNN_VMAX ? SNN_VMAX : v); }

// One LIF neuron update; returns 1 if it spiked. `in` is the already-summed synaptic input.
static inline int lif(int8_t &v, int in, int theta, int leak) {
    int x = v;
    x = x - asr(x, leak);
    x = sat(x + in);
    int z = x >= theta;
    if (z) x -= theta;
    v = (int8_t)x;
    return z;
}

void snn_reset(SnnState &st) {
#if SNN_H > 0
    for (int n = 0; n < SNN_H; n++) st.v1[n] = 0;
#endif
    for (int n = 0; n < SNN_OUT; n++) { st.v2[n] = 0; st.count[n] = 0; }
}

void snn_timestep(SnnState &st, const spikes_t &in_spikes) {
#if SNN_H > 0
    hidden_t h = 0;
    for (int n = 0; n < SNN_H; n++) {
        int in = dot_in(in_spikes, W1_POS[n], W1_NEG[n]);
        h |= (hidden_t)lif(st.v1[n], in, SNN_THETA1, SNN_LEAK1) << n;
    }
    for (int n = 0; n < SNN_OUT; n++) {
        int in = popcount32(h & W2_POS[n]) - popcount32(h & W2_NEG[n]);
        st.count[n] += (uint8_t)lif(st.v2[n], in, SNN_THETA2, SNN_LEAK2);
    }
#else
    for (int n = 0; n < SNN_OUT; n++) {
        int in = dot_in(in_spikes, W1_POS[n], W1_NEG[n]);
        st.count[n] += (uint8_t)lif(st.v2[n], in, SNN_THETA1, SNN_LEAK1);
    }
#endif
}

uint8_t snn_argmax(const SnnState &st) {
    uint8_t best = 0;
    for (int n = 1; n < SNN_OUT; n++)
        if (st.count[n] > st.count[best]) best = (uint8_t)n;  // strict '>' => lowest index wins ties
    return best;
}

uint8_t snn_infer(const spikes_t spikes[SNN_T], uint8_t counts_out[SNN_OUT]) {
    SnnState st;
    snn_reset(st);
    for (int t = 0; t < SNN_T; t++) snn_timestep(st, spikes[t]);
    for (int n = 0; n < SNN_OUT; n++) counts_out[n] = st.count[n];
    return snn_argmax(st);
}

uint8_t snn_step_top(const spikes_t &spike_word, bool clear, uint8_t counts_out[SNN_OUT]) {
#pragma HLS ARRAY_PARTITION variable=counts_out complete
    static SnnState st;  // persists between calls -> registers in RTL
    if (clear) snn_reset(st);
    snn_timestep(st, spike_word);
    for (int n = 0; n < SNN_OUT; n++) counts_out[n] = st.count[n];
    return snn_argmax(st);
}

void snn_rate_encode(const uint8_t levels[SNN_IN], spikes_t spikes[SNN_T]) {
    for (int t = 0; t < SNN_T; t++) {
        for (int k = 0; k < SNN_IW; k++) spikes[t].w[k] = 0;
        for (int p = 0; p < SNN_IN; p++) {
            int q = levels[p];
            if (((t + 1) * q) / 4 > (t * q) / 4) spikes[t].w[p / 64] |= 1ULL << (p % 64);
        }
    }
}
