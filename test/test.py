# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0
#
# cocotb tests for tt_um_nityashah1811_snn_mnist. Runs on the RTL (make) and on the sky130 gate-level
# netlist (make GATES=yes, done by the Tiny Tapeout gl_test action).
# Each image is sent through the real pin protocol and the predicted class plus all 10 spike
# counts are compared with the Python integer reference (test_vectors.txt).

import os
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

STB, CLR, DBG = 0x01, 0x02, 0x04
GATES = os.environ.get("GATES") == "yes"
N_IMAGES = int(os.environ.get("N_IMAGES", "3" if GATES else "12"))


def val(sig):
    v = sig.value
    try:
        return v.to_unsigned()  # cocotb >= 2.0
    except AttributeError:
        return int(v)


def load_vectors():
    vecs = []
    for line in (Path(__file__).parent / "test_vectors.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        vecs.append(([int(c) for c in f[0]], int(f[1]), int(f[2]), [int(x) for x in f[3:13]]))
    return vecs


def encode(levels):
    """Host-side deterministic rate code: 16 timesteps x 64-bit spike words."""
    words = []
    for t in range(16):
        w = 0
        for p, q in enumerate(levels):
            if ((t + 1) * q) // 4 > (t * q) // 4:
                w |= 1 << p
        words.append(w)
    return words


class Host:
    def __init__(self, dut):
        self.dut = dut
        self.uio = 0

    def set_uio(self, v):
        self.uio = v
        self.dut.uio_in.value = v

    def busy(self):
        return bool(val(self.dut.uo_out) & 0x20)

    async def wait_idle(self, limit=4000):
        for _ in range(limit // 8):
            if not self.busy():
                return
            await ClockCycles(self.dut.clk, 8)
        assert False, "timeout waiting for busy=0"

    async def reset(self):
        self.dut.rst_n.value = 0
        await ClockCycles(self.dut.clk, 10)
        self.dut.rst_n.value = 1
        await ClockCycles(self.dut.clk, 2)
        await self.wait_idle()

    async def clear(self):
        self.set_uio(self.uio | CLR)
        await ClockCycles(self.dut.clk, 4)
        self.set_uio(self.uio & ~CLR)
        await ClockCycles(self.dut.clk, 4)
        await self.wait_idle()

    async def load_byte(self, b):
        await self.wait_idle()
        self.dut.ui_in.value = b
        self.set_uio(self.uio | STB)
        await ClockCycles(self.dut.clk, 4)
        self.set_uio(self.uio & ~STB)
        await ClockCycles(self.dut.clk, 1)
        await self.wait_idle()

    async def send_image(self, levels, steps=16):
        words = encode(levels)
        for t in range(steps):
            for b in range(8):
                await self.load_byte((words[t] >> (8 * b)) & 0xFF)

    async def read_dbg(self, sel):
        self.set_uio((self.uio & ~0x7C) | DBG | (sel << 3))
        await ClockCycles(self.dut.clk, 12)
        v = val(self.dut.uo_out)
        self.set_uio(self.uio & ~0x7C)
        await ClockCycles(self.dut.clk, 1)
        return v


async def start(dut):
    cocotb.start_soon(Clock(dut.clk, 100, unit="ns").start())  # 10 MHz
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    host = Host(dut)
    await host.reset()
    return host


@cocotb.test()
async def test_alive_and_reset_state(dut):
    host = await start(dut)
    assert await host.read_dbg(15) == 0xA5, "debug word 15 must read 0xA5"
    assert await host.read_dbg(10) == 0, "step counter must be 0 after reset"
    status = val(dut.uo_out)
    assert (status & 0x0F) == 0 and (status & 0x30) == 0, f"bad status after reset: {status:#04x}"
    assert status & 0x40, "STEP_START (uo[6]) must be set after reset"
    assert val(dut.uio_oe) == 0 and val(dut.uio_out) == 0


@cocotb.test()
async def test_classify_images(dut):
    host = await start(dut)
    vecs = load_vectors()[:N_IMAGES]
    for i, (levels, label, gold, counts) in enumerate(vecs):
        await host.clear()
        await host.send_image(levels)
        status = val(dut.uo_out)
        assert status & 0x10, f"image {i}: DONE not set"
        assert (status & 0x0F) == gold, f"image {i}: predicted {status & 0x0F}, golden {gold}"
        for k in range(10):
            c = (await host.read_dbg(k)) & 0x1F
            assert c == counts[k], f"image {i}: count[{k}] = {c}, golden {counts[k]}"
        dut._log.info(f"image {i}: label {label}, predicted {gold}, counts {counts} OK")


@cocotb.test()
async def test_clear_mid_image_and_ignore_after_done(dut):
    host = await start(dut)
    (lv0, _, _, _), (lv1, _, gold1, _) = load_vectors()[:2]
    await host.send_image(lv0, steps=3)
    await host.load_byte(0x5A)
    await host.clear()
    assert await host.read_dbg(10) == 0 and await host.read_dbg(11) == 0, "clear must reset counters"
    await host.send_image(lv1)
    assert (val(dut.uo_out) & 0x1F) == (0x10 | gold1), "image 1 after mid-image clear"
    # bytes after DONE are ignored
    dut.ui_in.value = 0xFF
    host.set_uio(STB)
    await ClockCycles(dut.clk, 6)
    host.set_uio(0)
    await ClockCycles(dut.clk, 40)
    assert (val(dut.uo_out) & 0x3F) == (0x10 | gold1), "bytes after DONE must be ignored"
