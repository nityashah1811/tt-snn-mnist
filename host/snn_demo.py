# MicroPython for the Tiny Tapeout demo board (RP2040, tt-micropython-firmware).
# Copy to the board and run:  import snn_demo; snn_demo.run()
#
# NOTE: written against the tt-micropython-firmware DemoBoard API (DemoBoard.get(),
# shuttle.<project>.enable(), clock_project_PWM, reset_project, ui_in/uio_in/uo_out,
# uio_oe_pico). It has NOT been run on hardware -- check names against your firmware version.
import time
from ttboard.demoboard import DemoBoard

STB, CLR, DBG = 0x01, 0x02, 0x04

# First three images of test_vectors.txt: (64 levels 0-3, true label, expected chip prediction)
IMAGES = [
    ("0000000000000000011222100000020000000200000020000001100000030000", 7, 7),
    ("0002200000313000000030000003100000120000003000000023332200000000", 2, 2),
    ("0000000000000100000020000000200000001000000200000003000000000000", 1, 1),
]


def encode(levels):
    words = []
    for t in range(16):
        w = 0
        for p in range(64):
            q = levels[p]
            if ((t + 1) * q) // 4 > (t * q) // 4:
                w |= 1 << p
        words.append(w)
    return words


class SNNChip:
    def __init__(self, clock_hz=10_000_000):
        self.tt = DemoBoard.get()
        self.tt.shuttle.tt_um_nityashah1811_snn_mnist.enable()
        self.tt.uio_oe_pico.value = 0x7F  # RP2040 drives uio[6:0]
        self.tt.uio_in.value = 0
        self.tt.ui_in.value = 0
        self.tt.clock_project_PWM(clock_hz)
        self.tt.reset_project(True)
        time.sleep_ms(1)
        self.tt.reset_project(False)
        time.sleep_ms(1)

    def _uo(self):
        return int(self.tt.uo_out.value)

    def _wait_idle(self):
        for _ in range(10000):
            if not (self._uo() & 0x20):
                return
        raise RuntimeError("chip stuck busy")

    def _pulse(self, bit):
        self.tt.uio_in.value = bit
        time.sleep_us(5)          # >> 4 clocks at any clock >= 1 MHz
        self.tt.uio_in.value = 0
        time.sleep_us(5)

    def alive(self):
        self.tt.uio_in.value = DBG | (15 << 3)
        time.sleep_us(20)
        v = self._uo()
        self.tt.uio_in.value = 0
        return v == 0xA5

    def classify(self, levels):
        self._pulse(CLR)
        self._wait_idle()
        for w in encode(levels):
            for b in range(8):
                self.tt.ui_in.value = (w >> (8 * b)) & 0xFF
                self._pulse(STB)
                self._wait_idle()
        status = self._uo()
        assert status & 0x10, "DONE not set"
        return status & 0x0F

    def counts(self):
        out = []
        for k in range(10):
            self.tt.uio_in.value = DBG | (k << 3)
            time.sleep_us(20)
            out.append(self._uo() & 0x1F)
        self.tt.uio_in.value = 0
        return out


def run():
    chip = SNNChip()
    print("alive:", chip.alive())
    for lv, label, expected in IMAGES:
        levels = [int(c) for c in lv]
        p = chip.classify(levels)
        print("label", label, "-> predicted", p, "(expected", expected, ")",
              "OK" if p == expected else "MISMATCH", "counts", chip.counts())
