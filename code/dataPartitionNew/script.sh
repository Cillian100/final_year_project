printf "compiling main\n"
g++ -c main.cpp -o main.o -std=c++17
 
printf "compiling gpuCode\n"
nvcc -c ../sorting/gpuThrust.cu -o gpu.o
 
printf "compiling cpuCode\n"
g++ -c ../sorting/cpuRaduls.cpp -o cpu.o \
    -std=c++17 \
    -I. \
    -O3 \
    -DVERIFY_SORT

printf "compiling cpuIntel.cpp\n"
icpx -fPIE -c ../sorting/cpuIntel.cpp -o cpuIntel.o

 
printf "compiling RADULS\n"
g++ -c ../RADULS/Raduls/sorting_network.cpp -o sorting_network.o -std=c++17 -O1
 
printf "linking files\n"
g++ cpu.o gpu.o main.o cpuIntel.o sorting_network.o -o program \
    -L/usr/local/cuda/lib64 \
    -lcudart \
    -Wl,-rpath,/usr/local/cuda/lib64 \
    -lpthread
 
rm main.o gpu.o cpu.o sorting_network.o
 
./program
 
 