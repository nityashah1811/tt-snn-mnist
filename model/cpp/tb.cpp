// C simulation / co-simulation testbench. For every golden vector it runs BOTH
// snn_infer (whole image) and snn_step_top (the HLS top, one call per timestep) and checks
// the prediction AND all 10 spike counts match the Python integer reference exactly.
// Usage: tb [test_vectors.txt] [max_vectors]     exit code 0 = bit-exact on every vector
// (Use a small max_vectors for RTL co-simulation, e.g. 200, to keep it fast.)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "snn.h"

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "test_vectors.txt";
    int max_vectors = argc > 2 ? atoi(argv[2]) : 0;  // 0 = all
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }

    char line[1024], lv[256];
    int total = 0, correct = 0, mismatches = 0;
    while (fgets(line, sizeof line, f) && (max_vectors == 0 || total < max_vectors)) {
        if (line[0] == '#') continue;
        int label, gold, gc[SNN_OUT];
        if (sscanf(line, "%250s %d %d %d %d %d %d %d %d %d %d %d %d", lv, &label, &gold,
                   &gc[0], &gc[1], &gc[2], &gc[3], &gc[4], &gc[5], &gc[6], &gc[7], &gc[8], &gc[9]) != 13
            || (int)strlen(lv) != SNN_IN) {
            fprintf(stderr, "bad line %d\n", total + 1);
            return 2;
        }
        uint8_t levels[SNN_IN];
        for (int p = 0; p < SNN_IN; p++) levels[p] = (uint8_t)(lv[p] - '0');

        spikes_t spikes[SNN_T];
        snn_rate_encode(levels, spikes);

        uint8_t counts[SNN_OUT], step_counts[SNN_OUT];
        int pred = snn_infer(spikes, counts);
        int step_pred = 0;
        for (int t = 0; t < SNN_T; t++) step_pred = snn_step_top(spikes[t], t == 0, step_counts);

        bool ok = pred == gold && step_pred == gold;
        for (int n = 0; n < SNN_OUT; n++) ok &= counts[n] == gc[n] && step_counts[n] == gc[n];
        if (!ok && ++mismatches <= 5)
            fprintf(stderr, "MISMATCH vector %d: infer %d, step_top %d, golden %d\n", total, pred, step_pred, gold);
        correct += pred == label;
        total++;
    }
    fclose(f);
    printf("vectors: %d  bit-exact mismatches: %d  accuracy vs labels: %.2f%%\n",
           total, mismatches, 100.0 * correct / total);
    printf(mismatches == 0 ? "PASS\n" : "FAIL\n");
    return mismatches == 0 ? 0 : 1;
}
