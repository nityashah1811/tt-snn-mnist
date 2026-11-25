/*
 * Spiking neural network MNIST classifier for Tiny Tapeout (sky130, 1x1 tile).
 * SPDX-License-Identifier: Apache-2.0
 *
 * Network: 64 input spikes (8x8 image, rate-coded by the host) -> 10 LIF output neurons.
 * Ternary hardwired weights; T = 16 timesteps per image; class = output with most spikes.
 * Bit-exact with the Python/C++ reference (model/export.py, model/cpp/snn.cpp):
 *     v = sat8(v - (v >>> 4) + sum_i w_i * s_i);  spike = v >= 8;  if spike: v -= 8
 *     sat8 clamps to [-128, 127]; argmax ties -> lowest index.
 *
 * Architecture: the 10 neurons live in a shift-register ring that rotates EVERY clock
 * (10 x {5-bit spike count, 9-bit accumulator}). Only the entry passing the head is
 * modified, so the datapath is one small adder and there are no per-neuron enable muxes.
 * op tracks which neuron is at the head. The leak for the next timestep is applied when a
 * neuron is finalised (equivalent, since nothing happens to it in between), so the stored
 * value is always "leaked, ready to add". The 9-bit accumulator holds v + sum without
 * overflow (|v| <= 128, |sum| <= 64); saturation is applied once, at finalisation.
 *
 * Host protocol (all inputs may be asynchronous to clk; they are synchronised):
 *   CLEAR  : rising edge on uio[1] -> zero all neurons, counts, step and byte counters.
 *   LOAD   : put 8 spike bits on ui[7:0] (bit i = pixel 8*b+i, b = byte 0..7 of the step),
 *            then rising edge on uio[0]. Hold ui stable until busy (uo[5]) goes high or
 *            >= 4 clocks. Wait for busy = 0 before the next byte.
 *            After the 8th byte of a timestep the chip finalises that timestep automatically.
 *   RESULT : after 16 timesteps (128 bytes) done (uo[4]) = 1 and uo[3:0] = predicted class.
 *            Further bytes are ignored until CLEAR.
 *   DEBUG  : uio[2] = 1 switches uo[7:0] to the debug word selected by uio[6:3]:
 *            0..9 -> spike count of that class, 10 -> step counter, 11 -> byte index,
 *            15 -> 0xA5 (constant, "chip alive" check), others -> 0.
 */
`default_nettype none

module tt_um_nityashah1811_snn_mnist (
    input  wire [7:0] ui_in,    // spike byte
    output wire [7:0] uo_out,   // status / debug
    input  wire [7:0] uio_in,   // [0] strobe, [1] clear, [2] debug mode, [6:3] debug select
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

  // ------------------------------------------------------------------ synchronisers
  // Reset to all-ones so a pin that is already high at reset release is not seen as an edge.
  reg [2:0] stb_s, clr_s;
  always @(posedge clk) begin
    if (!rst_n) begin
      stb_s <= 3'b111;
      clr_s <= 3'b111;
    end else begin
      stb_s <= {stb_s[1:0], uio_in[0]};
      clr_s <= {clr_s[1:0], uio_in[1]};
    end
  end
  wire stb_rise = stb_s[1] & ~stb_s[2];
  wire clr_rise = clr_s[1] & ~clr_s[2];

  wire       dbg_mode = uio_in[2];
  wire [3:0] dbg_sel  = uio_in[6:3];

  // ------------------------------------------------------------------ state
  localparam [2:0] S_IDLE  = 3'd0,  // waiting for a byte
                   S_CLRW  = 3'd1,  // align to op == 0, then clear
                   S_CLR   = 3'd2,  // 10 clocks writing zeros into the ring
                   S_BYTEW = 3'd3,  // align to op == 0
                   S_BYTE  = 3'd4,  // 8 rotations: pixel pix of din into every neuron
                   S_FIN   = 3'd5;  // 1 rotation: saturate / fire / count / reset / leak

  reg  [2:0]   state;
  reg  [3:0]   op;        // neuron at ring head, free-running 0..9
  reg  [2:0]   pix;       // pixel within byte (S_BYTE)
  reg  [2:0]   bi;        // byte index within timestep
  reg  [4:0]   step;      // completed timesteps, 0..16
  reg  [7:0]   din;       // latched spike byte
  reg  [4:0]   best_cnt;
  reg  [3:0]   best_idx;
  reg  [3:0]   pred;
  reg  [4:0]   dbg_cnt;
  reg  [139:0] oring;     // 10 x {cnt[4:0], acc[8:0]}, entry at head = [13:0]

  wire op_last = (op == 4'd9);

  // ------------------------------------------------------------------ datapath
  wire [8:0] acc = oring[8:0];
  wire [4:0] cnt = oring[13:9];

  wire [1:0] w;
  snn_w u_w (.addr({op, bi, pix}), .w(w));
  wire [8:0] acc_add = acc + (din[pix] ? {{7{w[1]}}, w} : 9'd0);

  // sat8: clamp the 9-bit accumulator to [-128, 127]
  wire [7:0] v    = (acc[8] == acc[7]) ? acc[7:0] : {acc[8], {7{~acc[8]}}};
  wire       fire = ~v[7] & (|v[6:3]);                    // v >= 8
  wire [7:0] r    = fire ? (v - 8'd8) : v;                // subtractive reset
  wire [7:0] vn   = r - {{4{r[7]}}, r[7:4]};              // leak for the next timestep
  wire [4:0] cnt_new = cnt + {4'd0, fire};

  // running argmax over the ring (strict '>' => lowest index wins ties)
  wire       take   = (op == 4'd0) || (cnt_new > best_cnt);
  wire [4:0] nb_cnt = take ? cnt_new : best_cnt;
  wire [3:0] nb_idx = take ? op : best_idx;

  // ------------------------------------------------------------------ ring
  reg [13:0] o_tail;
  always @(*) begin
    case (state)
      S_CLR:   o_tail = 14'd0;
      S_BYTE:  o_tail = {cnt, acc_add};
      S_FIN:   o_tail = {cnt_new, vn[7], vn};
      default: o_tail = oring[13:0];  // entry passes through unchanged
    endcase
  end

  // Ring contents need no reset: the FSM always runs a clear pass after reset.
  always @(posedge clk) oring <= {o_tail, oring[139:14]};

  // ------------------------------------------------------------------ control
  always @(posedge clk) begin
    if (!rst_n) begin
      state    <= S_CLRW;
      op       <= 4'd0;
      pix      <= 3'd0;
      bi       <= 3'd0;
      step     <= 5'd0;
      din      <= 8'd0;
      best_cnt <= 5'd0;
      best_idx <= 4'd0;
      pred     <= 4'd0;
      dbg_cnt  <= 5'd0;
    end else begin
      op <= op_last ? 4'd0 : op + 4'd1;
      if (op == dbg_sel) dbg_cnt <= cnt;

      if (clr_rise) begin
        state <= S_CLRW;
      end else begin
        case (state)
          S_CLRW: if (op_last) state <= S_CLR;
          S_CLR: begin
            bi   <= 3'd0;
            step <= 5'd0;
            pred <= 4'd0;
            if (op_last) state <= S_IDLE;
          end
          S_IDLE: if (stb_rise && (step != 5'd16)) begin
            din   <= ui_in;
            state <= S_BYTEW;
          end
          S_BYTEW: if (op_last) begin
            pix   <= 3'd0;
            state <= S_BYTE;
          end
          S_BYTE: if (op_last) begin
            pix <= pix + 3'd1;
            if (pix == 3'd7) begin
              bi    <= bi + 3'd1;  // wraps 7 -> 0
              state <= (bi == 3'd7) ? S_FIN : S_IDLE;
            end
          end
          S_FIN: begin
            best_cnt <= nb_cnt;
            best_idx <= nb_idx;
            if (op_last) begin
              pred  <= nb_idx;
              step  <= step + 5'd1;
              state <= S_IDLE;
            end
          end
          default: state <= S_CLRW;
        endcase
      end
    end
  end

  // ------------------------------------------------------------------ outputs
  wire busy = (state != S_IDLE);
  wire done = (step == 5'd16);

  reg [7:0] dbg_word;
  always @(*) begin
    case (dbg_sel)
      4'd10:   dbg_word = {3'd0, step};
      4'd11:   dbg_word = {5'd0, bi};
      4'd15:   dbg_word = 8'hA5;
      default: dbg_word = (dbg_sel < 4'd10) ? {3'd0, dbg_cnt} : 8'd0;
    endcase
  end

  assign uo_out  = dbg_mode ? dbg_word : {1'b0, (bi == 3'd0), busy, done, pred};
  assign uio_out = 8'd0;
  assign uio_oe  = 8'd0;  // all bidirectional pins are inputs

  wire _unused = &{ena, uio_in[7], 1'b0};

endmodule
