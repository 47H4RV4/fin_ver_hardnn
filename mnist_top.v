`timescale 1ns / 1ps

module mnist_top (
    input clk,
    input rst,
    input start,
    output [3:0] predicted_class,
    output done
);
    // Control & Address signals
    wire [31:0] internal_addr;
    wire l1_run, l2_run, l3_run;
    wire [127:0] l1_valids; wire [31:0] l2_valids; wire [9:0] l3_valids;
    
    // 4-bit Parallel Buses
    wire [128*4-1:0] l1_bus; 
    wire [32*4-1:0] l2_bus; 
    wire [10*4-1:0] l3_bus;
    
    wire [3:0] rom_pixel;

    global_controller ctrl (
        .clk(clk), .rst(rst), .start_network(start),
        .l1_done(l1_valids[0]), .l2_done(l2_valids[0]), .l3_done(l3_valids[0]),
        .current_addr(internal_addr), .l1_run(l1_run), .l2_run(l2_run), .l3_run(l3_run),
        .network_ready()
    );

    // Image Input
    image_rom img (.clk(clk), .addr(internal_addr[9:0]), .q(rom_pixel));

    // Layer 1: 784 -> 128 (Int4, Shift 6)
    nn_layer #(.NUM_INPUTS(784), .NUM_NEURONS(128), .SHIFT(6), .WEIGHT_FILE("w1.mem")) L1 (
        .clk(clk), .rst(rst), .data_in(rom_pixel), .input_valid(l1_run),
        .local_addr(internal_addr), .out_valids(l1_valids), .layer_out(l1_bus)
    );

    // Layer 2: 128 -> 32 (Int4, Shift 6)
    // Select correct 4-bit chunk from L1 bus based on current address
    nn_layer #(.NUM_INPUTS(128), .NUM_NEURONS(32), .SHIFT(6), .WEIGHT_FILE("w2.mem")) L2 (
        .clk(clk), .rst(rst), .data_in(l1_bus[internal_addr[6:0]*4 +: 4]), .input_valid(l2_run),
        .local_addr(internal_addr), .out_valids(l2_valids), .layer_out(l2_bus)
    );

    // Layer 3: 32 -> 10 (Int4, Shift 6)
    // Select correct 4-bit chunk from L2 bus
    nn_layer #(.NUM_INPUTS(32), .NUM_NEURONS(10), .SHIFT(6), .WEIGHT_FILE("w3.mem")) L3 (
        .clk(clk), .rst(rst), .data_in(l2_bus[internal_addr[4:0]*4 +: 4]), .input_valid(l3_run),
        .local_addr(internal_addr), .out_valids(l3_valids), .layer_out(l3_bus)
    );

    // Classification
    argmax_unit am (
        .clk(clk), .rst(rst), .in_valid(l3_valids[0]),
        .neuron_outputs(l3_bus), .prediction(predicted_class), .out_valid(done)
    );
endmodule