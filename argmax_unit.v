`timescale 1ns / 1ps

module argmax_unit (
    input clk,
    input rst,
    input in_valid,
    input [10*4-1:0] neuron_outputs, // Parallel 4-bit logits
    output reg [3:0] prediction,
    output reg out_valid
);
    integer i;
    reg [3:0] max_val;

    always @(posedge clk) begin
        if (rst) begin
            prediction <= 0;
            out_valid <= 0;
            max_val <= 0;
        end else if (in_valid) begin
            // Hardware comparator to find the winning digit
            max_val = 4'd0;
            prediction = 4'd0;
            for (i = 0; i < 10; i = i + 1) begin
                if (neuron_outputs[i*4 +: 4] > max_val) begin
                    max_val = neuron_outputs[i*4 +: 4];
                    prediction = i[3:0];
                end
            end
            out_valid <= 1;
        end else begin
            out_valid <= 0;
        end
    end
endmodule