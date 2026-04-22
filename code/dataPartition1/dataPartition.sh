

printf "compiling main.cpp\n"
icpx -c main.cpp -o main.o

printf "compiling cpuCode.cpp\n"
icpx -c cpuCode.cpp -o cpu.o

printf "compiling gpuCode.cu\n"
nvcc -c gpuCode.cu -o gpu.o

printf "linking files\n"
icpx cpu.o gpu.o main.o -o program \
    -ltbb \
    -L/usr/local/cuda/lib64 \
    -lcudart \
    -Wl,-rpath,/usr/local/cuda/lib64

rm main.o
rm cpu.o
rm gpu.o

./program

##python3 test.py