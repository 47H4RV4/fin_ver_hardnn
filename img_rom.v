`timescale 1ns / 1ps

module image_rom (
    input clk,
    input [9:0] addr,
    output reg [3:0] q
);
    reg [3:0] rom [0:783];
    initial $readmemh("input1.mem", rom); // Updated to hex
    always @(posedge clk) q <= rom[addr];
endmodule