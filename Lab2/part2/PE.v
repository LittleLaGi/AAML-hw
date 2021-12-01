module PE(clk, reset_n, enable, weight, up_i, left_i, right_o, down_o);
    
input wire clk;
input wire reset_n;
input wire enable;
input wire [7:0]weight;
input wire [7:0]left_i;
input wire [15:0]up_i;
output reg [7:0]right_o;
output reg [15:0]down_o;

always@(posedge clk) begin

    if (~enable) begin
        down_o <= 16'b0;
        right_o <= 8'b0;
    end
    
    else begin
        down_o <= weight * left_i + up_i;
        right_o <= left_i;
    end

end
    
endmodule