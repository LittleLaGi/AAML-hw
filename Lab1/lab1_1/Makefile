
no_rvv:
	~/RISC/llvm-project/build/bin/clang --target=riscv64-unknown-elf -menable-experimental-extensions -march=rv64gcv1p0 --sysroot=/home/littlelagi/RISC/riscv-gnu-toolchain_rvv-0.9.x/riscv64-unknown-elf --gcc-toolchain=/home/littlelagi/RISC/riscv-gnu-toolchain_rvv-0.9.x -o saxpy_no_rvv saxpy_no_rvv.c saxpy_no_rvv.s
	riscv64-unknown-elf-objdump -d saxpy_no_rvv > saxpy_no_rvv.dump
	spike -d pk -s saxpy_no_rvv

rvv:
	~/RISC/llvm-project/build/bin/clang --target=riscv64-unknown-elf -menable-experimental-extensions -march=rv64gcv1p0 --sysroot=/home/littlelagi/RISC/riscv-gnu-toolchain_rvv-0.9.x/riscv64-unknown-elf --gcc-toolchain=/home/littlelagi/RISC/riscv-gnu-toolchain_rvv-0.9.x -o saxpy_rvv saxpy_rvv.c saxpy_rvv.s
	riscv64-unknown-elf-objdump -d saxpy_rvv > saxpy_rvv.dump
	spike -d pk -s saxpy_rvv

clean:
	rm saxpy_rvv saxpy_no_rvv