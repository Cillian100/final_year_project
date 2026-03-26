ICPX=/opt/intel/oneapi/compiler/2023.0.0/linux/bin/icpx
ONEDPL_INCLUDE=/opt/intel/oneapi/dpl/2022.0.0/include


printf "compiling measuringThrustSpeed.cu\n"
nvcc -c code/measuringThrustSpeed.cu -o measuringThrust.o

printf "compiling measuringTbbSpeed.cpp\n"
$ICPX -std=c++17 -O2 -c code/measuringTbbSpeed.cpp -o measuringTbb.o \
    -I$ONEDPL_INCLUDE \
    -I$HOME/onedpl/include \
    -I$HOME/tbb-local/include \
    -L$HOME/tbb-local/lib \
    -ltbb -Wl,-rpath,$HOME/tbb-local/lib


printf "linking files\n"
$ICPX measuringThrust.o measuringTbb.o -o program \
	-I$ONEDPL_INCLUDE \
	-I$HOME/onedpl/include \
	-I$HOME/tbb-local/include \
	-L$HOME/tbb-local/lib \
   	-ltbb -Wl,-rpath,$HOME/tbb-local/lib -lcudart -L/usr/local/cuda/lib64


rm measuringThrust.o
rm measuringTbb.o

./program

rm program