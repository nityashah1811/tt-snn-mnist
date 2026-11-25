# Vitis HLS flow: C simulation -> C synthesis -> C/RTL co-simulation.
# Run from THIS folder:
#   vitis_hls -f run_hls.tcl                     (Vitis HLS 2024.1 and earlier)
#   vitis-run --mode hls --tcl run_hls.tcl       (Vitis 2024.2 and later)
# Generated Verilog lands in snn_prj/sol1/syn/verilog/

set vec [file normalize ../test_vectors.txt]

open_project -reset snn_prj
set_top snn_step_top
add_files snn.cpp
add_files snn.h
add_files snn_weights.h
add_files -tb tb.cpp
add_files -tb ../test_vectors.txt

open_solution -reset sol1 -flow_target vivado
# Vitis HLS needs an FPGA part even though the RTL is headed for sky130; any small part works.
set_part xc7a35tcpg236-1
# 50 MHz, the usual Tiny Tapeout clock
create_clock -period 20 -name default

csim_design -argv "$vec"
csynth_design
# RTL co-sim on the first 200 images (each is 16 calls); raise it if you have time.
cosim_design -argv "$vec 200"
exit
