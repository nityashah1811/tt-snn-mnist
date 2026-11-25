#!/usr/bin/env bash
# Pin-level testbench (Yosys CXXRTL) on (1) the RTL and (2) the Yosys gate netlist.
# Needs: pip install yowasp-yosys ziglang   (or set CXX to any C++14 compiler)
# Usage: sim/run_sim.sh [max_images_rtl] [max_images_gl]   (default: all 10000 / 2000)
set -euo pipefail
cd "$(dirname "$0")/.."
N_RTL=${1:-0}
N_GL=${2:-2000}
VEC=model/test_vectors.txt
CXX=${CXX:-"python -m ziglang c++"}
INC=$(python -c "import yowasp_yosys,os;print(os.path.join(os.path.dirname(yowasp_yosys.__file__),'share','include','backends','cxxrtl','runtime'))")
TOP=tt_um_nityashah1811_snn_mnist
mkdir -p sim/build
SRCS="src/project.v src/snn_weights.v"

echo "== RTL -> CXXRTL"
yowasp-yosys -q -p "read_verilog $SRCS; hierarchy -top $TOP; write_cxxrtl sim/build/sim_rtl.cc"
$CXX -std=c++14 -O2 -I"$INC" -DSIM_SOURCE='"build/sim_rtl.cc"' sim/tb_cxxrtl.cpp -o sim/build/tb_rtl.exe
./sim/build/tb_rtl.exe "$VEC" "$N_RTL"

echo "== Gate netlist (yosys synth) -> CXXRTL"
yowasp-yosys -q -p "read_verilog $SRCS; synth -flatten -top $TOP; write_verilog -noattr sim/build/gl_generic.v"
yowasp-yosys -q -p "read_verilog sim/build/gl_generic.v; hierarchy -top $TOP; write_cxxrtl sim/build/sim_gl.cc"
$CXX -std=c++14 -O2 -I"$INC" -DSIM_SOURCE='"build/sim_gl.cc"' sim/tb_cxxrtl.cpp -o sim/build/tb_gl.exe
./sim/build/tb_gl.exe "$VEC" "$N_GL"
