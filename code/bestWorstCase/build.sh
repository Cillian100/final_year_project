printf "compiling main\n"
g++ -c main.cpp -o main.o -std=c++17
 
printf "comping gpuCode\n"
nvcc -c gpuCode.cu -o gpu.o
 
printf "compiling cpuCode\n"
g++ -c cpuCode.cpp -o cpu.o \
    -std=c++17 \
    -I. \
    -O3
 
printf "compiling RADULS\n"
g++ -c ../RADULS/Raduls/sorting_network.cpp -o sorting_network.o -std=c++17 -O1
 
printf "linking files\n"
g++ cpu.o gpu.o main.o sorting_network.o -o program \
    -L/usr/local/cuda/lib64 \
    -lcudart \
    -Wl,-rpath,/usr/local/cuda/lib64 \
    -lpthread
 
rm main.o gpu.o cpu.o sorting_network.o
 
./program
 
 