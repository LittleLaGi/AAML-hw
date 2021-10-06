#!/bin/bash

riscv64-unknown-elf-gcc saxpy_no_rvv.c saxpy_no_rvv.s -march=rv64gcv -o saxpy_no_rvv
spike pk -s saxpy_no_rvv
