`timescale 1ns / 1ps

module mnist_tb;
    // Testbench Signals
    reg clk;
    reg rst;
    reg start;
    wire [3:0] predicted_class;
    wire done;

    // Instantiate Top Module
    mnist_top uut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .predicted_class(predicted_class),
        .done(done)
    );

    // Clock Generation (100MHz)
    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        // Initialize and Reset
        $display("[%0t] TB: Initializing System...", $time);
        rst = 1;
        start = 0;
        #100;
        rst = 0;
        #20;
        
        // Trigger Inference
        $display("[%0t] TB: Sending Start Signal", $time);
        start = 1;
        #10;
        start = 0;

        // Monitor for Completion with Timeout
        fork : timeout_block
            begin
                wait(done);
                $display("\n-----------------------------------------");
                $display("INFERENCE SUCCESSFUL");
                $display("Hardware Predicted Digit: %d", predicted_class);
                $display("-----------------------------------------\n");
                disable timeout_block;
            end
            begin
                #20000000; // 20ms safety timeout
                $display("\n[ERROR] TB: Inference Timeout! Signal 'done' never received.");
                disable timeout_block;
            end
        join

        #100;
        $finish;
    end

    // Waveform Logging
    initial begin
        $dumpfile("mnist_inference.vcd");
        $dumpvars(0, mnist_tb);
    end

endmodule