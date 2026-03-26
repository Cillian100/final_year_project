ICPX=/opt/intel/oneapi/compiler/2023.0.0/linux/bin/icpx
ONEDPL_INCLUDE=/opt/intel/oneapi/dpl/2022.0.0/include
#source /opt/intel/oneapi/setvars.sh

printf "compiling main.cpp\n"
g++ -c main.cpp -o main.o

printf "compiling gpu_sort.cu\n"
nvcc -c gpu_sort.cu -o gpu_sort.o

printf "compiling cpu_sort.cu\n"
$ICPX -c -std=c++17 -O2 cpu_sort.cpp -o cpu_sort.o \
    -I$ONEDPL_INCLUDE \
    -I$HOME/onedpl/include \
    -I$HOME/tbb-local/include \
    -L$HOME/tbb-local/lib \
    -ltbb -Wl,-rpath,$HOME/tbb-local/lib

printf "linking files\n"
$ICPX main.o gpu_sort.o cpu_sort.o -o ~/exe/program \
	-I$ONEDPL_INCLUDE \
	-I$HOME/onedpl/include \
	-I$HOME/tbb-local/include \
	-L$HOME/tbb-local/lib \
   	-ltbb -Wl,-rpath,$HOME/tbb-local/lib -lcudart -L/usr/local/cuda/lib64


rm gpu_sort.o

rm main.o

rm cpu_sort.o

#cd executable

printf "running program\n"
./program 

#cd ..
