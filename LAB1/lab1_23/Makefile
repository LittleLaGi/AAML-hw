##################################################################
# To run the make, you should have the following tools available:
#
#   * rvv clang/llvm (https://github.com/isrc-cas/rvv-llvm)
#   * riscv gcc elf toolchain (https://github.com/riscv/riscv-gnu-toolchain)
#   * spike (https://github.com/riscv/riscv-isa-sim)
#   * pk (https://github.com/riscv/riscv-pk)
#
##################################################################

# Change the path for your environment.

GCC_TOOLCHAIN_DIR := /home/littlelagi/RISC/riscv-gnu-toolchain_rvv-0.9.x
SYSROOT_DIR := /home/littlelagi/RISC/riscv-gnu-toolchain_rvv-0.9.x/riscv64-unknown-elf

CLANG := ~/RISC/llvm-project/build/bin/clang
SPIKE := spike
PK := pk

objects = mtmac.c mtmac.s
target = mtmac-test

$(target) : $(objects)
	$(CLANG)  --target=riscv64-unknown-elf -menable-experimental-extensions -march=rv64gcv1p0 --sysroot=$(SYSROOT_DIR) --gcc-toolchain=$(GCC_TOOLCHAIN_DIR) -o $(target) $(objects)

run :
	$(SPIKE) $(PK) -s $(target)

clean :
	rm -f $(target) mtmtp mtmac

mtp:
	~/RISC/llvm-project/build/bin/clang  --target=riscv64-unknown-elf -menable-experimental-extensions -march=rv64gcv1p0 --sysroot=/home/littlelagi/RISC/riscv-gnu-toolchain_rvv-0.9.x/riscv64-unknown-elf --gcc-toolchain=/home/littlelagi/RISC/riscv-gnu-toolchain_rvv-0.9.x -o mtmtp mtmtp.c mtmtp.s
	riscv64-unknown-elf-objdump -d mtmtp > mtmtp.dump
	spike -d --isa=rv64gcv  /home/littlelagi/RISC/riscv-gnu-toolchain_rvv-0.9.x/riscv64-unknown-elf/bin/pk -s mtmtp

mac:
	~/RISC/llvm-project/build/bin/clang  --target=riscv64-unknown-elf -menable-experimental-extensions -march=rv64gcv1p0 --sysroot=/home/littlelagi/RISC/riscv-gnu-toolchain_rvv-0.9.x/riscv64-unknown-elf --gcc-toolchain=/home/littlelagi/RISC/riscv-gnu-toolchain_rvv-0.9.x -o mtmac mtmac.c mtmac.s
	riscv64-unknown-elf-objdump -d mtmac > mtmac.dump
	spike -d --isa=rv64gcv  /home/littlelagi/RISC/riscv-gnu-toolchain_rvv-0.9.x/riscv64-unknown-elf/bin/pk -s mtmac