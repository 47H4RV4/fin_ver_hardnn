`timescale 1ns / 1ps

module Weight_Memory #(
    parameter NEURON_ID = 0,
    parameter NUM_WEIGHTS_PER_NEURON = 784,
    parameter TOTAL_WORDS = 100352, // e.g., 784 * 128
    parameter WEIGHT_FILE = "w1.mem"
)(
    input clk,
    input [31:0] local_addr,
    output reg [3:0] weight_out
);

    // Memory sized for the entire giant file
    reg [3:0] mem [0:TOTAL_WORDS-1];

    initial begin
        // Use $readmemh for hex files from img_maker.py
        $readmemh(WEIGHT_FILE, mem);
    end

    always @(posedge clk) begin
        // Offset addressing: (Neuron Index * Weights Per Neuron) + Current Address
        weight_out <= mem[(NEURON_ID * NUM_WEIGHTS_PER_NEURON) + local_addr];
    end
endmodule