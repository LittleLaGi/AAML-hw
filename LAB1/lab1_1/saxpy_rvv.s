    .text
    .balign 4
    .global saxpy_rvv
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

# Please finish this RISC-V V extension code.
saxpy_rvv:
    vsetvli a4, a0, e32, m8, ta,ma
    #vle32.v 
    #sub 
    #slli 
    #add 
    #vle32.v
    #vfmacc.vf
    #vse32.v
    #add 
    bnez a0, saxpy_rvv
    ret
