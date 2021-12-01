`include "PE.v"

module Systolic_array(clk, reset_n, enable, weight_i, input_i, result_o);

input wire clk, reset_n, enable;
input wire [8*16*16-1:0] weight_i;
input wire [8*16-1:0] input_i;
output wire [16*16-1:0] result_o;

wire[15:0] up_wire[0:15][0:15];
wire[15:0] down_wire[0:15][0:15];
wire[7:0] left_wire[0:15][0:15];
wire[7:0] right_wire[0:15][0:15];

genvar up_iter;
generate
for (up_iter = 0; up_iter < 16; up_iter = up_iter + 1) begin: init_up_wire
    assign up_wire[0][up_iter] = 16'b0;
end
endgenerate

genvar down_iter;
generate
for (down_iter = 0; down_iter < 16; down_iter = down_iter + 1) begin: init_down_wire
    assign result_o[down_iter * 16 + 15 : down_iter * 16] = down_wire[15][down_iter];
end
endgenerate

genvar left_iter;
generate
for (left_iter = 0; left_iter < 16; left_iter = left_iter + 1) begin: init_left_wire
    assign left_wire[left_iter][0] = input_i[left_iter * 8 + 7 : left_iter * 8];
end
endgenerate


genvar i;
genvar j;
generate
for (i = 0; i < 16; i = i + 1) begin: outer
    for (j = 0; j < 16; j = j + 1) begin: inner
        PE  pe (
            .clk(clk),
            .reset_n(reset_n),
            .enable(enable),
            .weight(weight_i[(i * 16 + j) * 8 + 7 : (i * 16 + j) * 8]),
            .up_i(up_wire[i][j]),
            .left_i(left_wire[i][j]),
            .right_o(right_wire[i][j]),
            .down_o(down_wire[i][j])
        );
        
        if (i > 0) begin
            assign up_wire[i][j] = down_wire[i - 1][j];
        end
        
        if (j > 0) begin
            assign left_wire[i][j] = right_wire[i][j - 1];
        end
    end
end
endgenerate

endmodule