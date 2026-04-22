printf "compiling main\n"
icpx -c main.cpp -o main.o

printf "comping gpuCode\n"
nvcc -c gpuCode.cu -o gpu.o

printf "compiling cpuCode\n"
icpx -c cpuCode.cpp -o cpu.o

printf "linking files\n"
icpx cpu.o gpu.o main.o -o program \
    -ltbb \
    -L/usr/local/cuda/lib64 \
    -lcudart \
    -Wl,-rpath,/usr/local/cuda/lib64

rm main.o
rm gpu.o

./program