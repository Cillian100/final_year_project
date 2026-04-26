#!/bin/bash
 
needs_rebuild() {
    local src=$1
    local obj=$2
    [ ! -f "$obj" ] || [ "$src" -nt "$obj" ]
}
 
if needs_rebuild main.cpp main.o; then
    printf "compiling main\n"
    g++ -c main.cpp -o main.o -std=c++17 || exit 1
fi
 
if needs_rebuild gpuCode.cu gpu.o; then
    printf "compiling gpuCode\n"
    nvcc -c gpuCode.cu -o gpu.o || exit 1
fi
 
if needs_rebuild cpuCode.cpp cpu.o; then
    printf "compiling cpuCode\n"
    g++ -c cpuCode.cpp -o cpu.o \
        -std=c++17 \
        -I. \
        -O3 || exit 1
fi
 
if needs_rebuild ../RADULS/Raduls/sorting_network.cpp sorting_network.o; then
    printf "compiling RADULS\n"
    g++ -c ../RADULS/Raduls/sorting_network.cpp -o sorting_network.o -std=c++17 -O1 || exit 1
fi
 
if needs_rebuild main.o program || needs_rebuild gpu.o program || \
   needs_rebuild cpu.o program  || needs_rebuild sorting_network.o program; then
    printf "linking files\n"
    g++ cpu.o gpu.o main.o sorting_network.o -o program \
        -ltbb \
        -L/usr/local/cuda/lib64 \
        -lcudart \
        -Wl,-rpath,/usr/local/cuda/lib64 \
        -lpthread || exit 1
fi
 
 printf "running program\n"
./program