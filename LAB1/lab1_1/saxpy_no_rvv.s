    .text
    .balign 4
    .global saxpy_no_rvv
# void
# saxpy(size_t n, const float a, const float *x, float *y)
# {
#   size_t i;
#   for (i=0; i<n; i++)
#     y[i] = a * x[i] + y[i];
# }
#
# register arguments:
#     a0      n
#     fa0     a
#     a1      x
#     a2      y

# Please finish this RISC-V code.
saxpy_no_rvv:
    add t6, zero, zero
saxpy_L1:
    add t3, t6, zero
    slli t3, t3, 2
    add t4, a1, t3
    flw ft0, 0(t4)
    add t4, a2, t3
    flw ft1, 0(t4)
    fmul.s ft0, ft0, fa0
    fadd.s ft0, ft0, ft1
    fsw ft0, 0(t4)
    addi t6, t6, 1
	blt t6, a0, saxpy_L1
    ret
