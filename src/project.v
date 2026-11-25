/*
 * Spiking neural network MNIST classifier for Tiny Tapeout (sky130).
 * SPDX-License-Identifier: Apache-2.0
 *
 * Network: 64 input spikes (8x8 image, rate-coded by the host) -> 24 LIF -> 10 LIF.
 * Ternary hardwired weights; T = 16 timesteps per image; class = output with most spikes.
 * Bit-exact with the Python/C++ reference (export.py / hls/snn.cpp):
 *     v = sat6(v - (v >>> 4) + sum_i w_i * s_i);  spike = v >= THETA;  if spike: v -= THETA
 *     THETA = 8 (hidden), 4 (output); sat6 clamps to [-32, 31]; argmax ties -> lowest index.
 *
 * Architecture: neuron state lives in two shift-register rings that rotate EVERY clock
 * (hidden: 24 x 8-bit accumulators, output: 10 x {5-bit count, 7-bit accumulator}).
 * Only the entry passing the head is modified, so the whole datapath is one tiny adder per
 * ring and there are no per-neuron enable muxes. hp / op track which neuron is at the head.
 * The leak for the next timestep is applied when a neuron is finalised (equivalent, since
 * nothing happens to it in between), so the stored value is always "leaked, ready to add".
 *
 * Host protocol (all inputs may be asynchronous to clk; they are synchronised):
 *   CLEAR  : rising edge on uio[1] -> zero all neurons, counts, step and byte counters.
 *   LOAD   : put 8 spike bits on ui[7:0] (bit i = pixel 8*b+i, b = byte 0..7 of the step),
 *            then rising edge on uio[0]. Hold ui stable until busy (uo[5]) goes high or
 *            >= 4 clocks. Wait for busy = 0 before the next byte.
 *            After the 8th byte of a timestep the network runs that timestep automatically.
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
  localparam [3:0] S_IDLE  = 4'd0,  // waiting for a byte
                   S_CLRW  = 4'd1,  // align to hp == 0, then clear
                   S_CLR   = 4'd2,  // 24 clocks writing zeros into both rings
                   S_BYTEW = 4'd3,  // align to hp == 0
                   S_BYTE  = 4'd4,  // 8 rotations: pixel pix of din into every hidden neuron
                   S_FIN1  = 4'd5,  // 1 rotation: saturate / fire / reset / leak hidden layer
                   S_L2W   = 4'd6,  // align to op == 0
                   S_L2    = 4'd7,  // 24 rotations: hidden spike hj into every output neuron
                   S_FIN2  = 4'd8;  // 1 rotation: finalise outputs, count spikes, argmax

  reg  [3:0]   state;
  reg  [4:0]   hp;        // hidden neuron at ring head, free-running 0..23
  reg  [3:0]   op;        // output neuron at ring head, free-running 0..9
  reg  [2:0]   pix;       // pixel within byte          (S_BYTE)
  reg  [4:0]   hj;        // hidden neuron being routed (S_L2)
  reg  [2:0]   bi;        // byte index within timestep
  reg  [4:0]   step;      // completed timesteps, 0..16
  reg  [7:0]   din;       // latched spike byte
  reg  [23:0]  hspk;      // hidden spikes of the current timestep, bit 0 = next to route
  reg  [4:0]   best_cnt;
  reg  [3:0]   best_idx;
  reg  [3:0]   pred;
  reg  [4:0]   dbg_cnt;
  reg  [191:0] hring;     // 24 x 8-bit accumulator, entry at head = [7:0]
  reg  [119:0] oring;     // 10 x {cnt[4:0], acc[6:0]}, entry at head = [11:0]

  wire hp_last = (hp == 5'd23);
  wire op_last = (op == 4'd9);

  // ------------------------------------------------------------------ helpers
  // Clamp an 8-bit two's-complement value to the 6-bit range [-32, 31].
  function [5:0] sat6;
    input [7:0] x;
    begin
      if (!x[7] && (x[6:5] != 2'b00))     sat6 = 6'b011111;  // > 31
      else if (x[7] && (x[6:5] != 2'b11)) sat6 = 6'b100000;  // < -32
      else                                sat6 = x[5:0];
    end
  endfunction

  // Subtractive reset, then leak for the next timestep: r = x - fire*theta; r - (r >>> 4).
  function [5:0] lif_next;
    input [5:0] x;
    input       fire;
    input [4:0] theta;
    reg   [5:0] r;
    begin
      r        = fire ? (x - {1'b0, theta}) : x;
      lif_next = r - {{4{r[5]}}, r[5:4]};
    end
  endfunction

  // ------------------------------------------------------------------ hidden datapath
  wire [7:0] h_head = hring[7:0];
  wire [1:0] w1;
  snn_w1 u_w1 (.addr({hp, bi, pix}), .w(w1));
  wire [7:0] h_add  = h_head + (din[pix] ? {{6{w1[1]}}, w1} : 8'd0);
  wire [5:0] h_sat  = sat6(h_head);
  wire       h_fire = ~h_sat[5] & (h_sat[4:0] >= 5'd8);
  wire [5:0] h_new  = lif_next(h_sat, h_fire, 5'd8);

  // ------------------------------------------------------------------ output datapath
  wire [6:0] o_acc = oring[6:0];
  wire [4:0] o_cnt = oring[11:7];
  wire [1:0] w2;
  snn_w2 u_w2 (.addr({hj, op}), .w(w2));
  wire [6:0] o_add   = o_acc + (hspk[0] ? {{5{w2[1]}}, w2} : 7'd0);
  wire [5:0] o_sat   = sat6({o_acc[6], o_acc});
  wire       o_fire  = ~o_sat[5] & (o_sat[4:0] >= 5'd4);
  wire [5:0] o_new   = lif_next(o_sat, o_fire, 5'd4);
  wire [4:0] cnt_new = o_cnt + {4'd0, o_fire};

  // running argmax over the output ring (strict '>' => lowest index wins ties)
  wire       take   = (op == 4'd0) || (cnt_new > best_cnt);
  wire [4:0] nb_cnt = take ? cnt_new : best_cnt;
  wire [3:0] nb_idx = take ? op : best_idx;

  // ------------------------------------------------------------------ rings
  reg [7:0]  h_tail;
  reg [11:0] o_tail;
  always @(*) begin
    h_tail = h_head;          // default: entry passes through unchanged
    o_tail = oring[11:0];
    case (state)
      S_CLR: begin
        h_tail = 8'd0;
        o_tail = 12'd0;
      end
      S_BYTE: h_tail = h_add;
      S_FIN1: h_tail = {{2{h_new[5]}}, h_new};
      S_L2:   o_tail = {o_cnt, o_add};
      S_FIN2: o_tail = {cnt_new, o_new[5], o_new};
      default: ;
    endcase
  end

  // Ring contents need no reset: the FSM always runs a clear pass after reset.
  always @(posedge clk) begin
    hring <= {h_tail, hring[191:8]};
    oring <= {o_tail, oring[119:12]};
  end

  // ------------------------------------------------------------------ control
  always @(posedge clk) begin
    if (!rst_n) begin
      state    <= S_CLRW;
      hp       <= 5'd0;
      op       <= 4'd0;
      pix      <= 3'd0;
      hj       <= 5'd0;
      bi       <= 3'd0;
      step     <= 5'd0;
      din      <= 8'd0;
      hspk     <= 24'd0;
      best_cnt <= 5'd0;
      best_idx <= 4'd0;
      pred     <= 4'd0;
      dbg_cnt  <= 5'd0;
    end else begin
      hp <= hp_last ? 5'd0 : hp + 5'd1;
      op <= op_last ? 4'd0 : op + 4'd1;
      if (op == dbg_sel) dbg_cnt <= o_cnt;

      if (clr_rise) begin
        state <= S_CLRW;
      end else begin
        case (state)
          S_CLRW: if (hp_last) state <= S_CLR;
          S_CLR: begin
            bi   <= 3'd0;
            step <= 5'd0;
            pred <= 4'd0;
            if (hp_last) state <= S_IDLE;
          end
          S_IDLE: if (stb_rise && (step != 5'd16)) begin
            din   <= ui_in;
            state <= S_BYTEW;
          end
          S_BYTEW: if (hp_last) begin
            pix   <= 3'd0;
            state <= S_BYTE;
          end
          S_BYTE: if (hp_last) begin
            pix <= pix + 3'd1;
            if (pix == 3'd7) begin
              bi    <= bi + 3'd1;  // wraps 7 -> 0
              state <= (bi == 3'd7) ? S_FIN1 : S_IDLE;
            end
          end
          S_FIN1: begin
            hspk <= {h_fire, hspk[23:1]};
            if (hp_last) state <= S_L2W;
          end
          S_L2W: if (op_last) begin
            hj    <= 5'd0;
            state <= S_L2;
          end
          S_L2: if (op_last) begin
            hspk <= {1'b0, hspk[23:1]};
            hj   <= hj + 5'd1;
            if (hj == 5'd23) state <= S_FIN2;
          end
          S_FIN2: begin
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
