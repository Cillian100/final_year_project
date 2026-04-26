CXX = g++
NVCC = nvcc
ICPX = icpx

CXXFLAGS = -std=c++17 -O3
CUDA_LIB = /usr/local/cuda/lib64

OBJS = main.o gpu.o cpu.o cpuIntel.o sorting_network.o

program: $(OBJS)
	$(ICPX) $(OBJS) -o program \
		-L$(CUDA_LIB) \
		-lcudart \
		-Wl,-rpath,$(CUDA_LIB) \
		-lpthread \
		-ltbb

main.o: main.cpp
	$(CXX) -c main.cpp -o main.o $(CXXFLAGS)

gpu.o: ../sorting/gpuThrust.cu
	$(NVCC) -c ../sorting/gpuThrust.cu -o gpu.o

cpu.o: ../sorting/cpuRaduls.cpp
	$(CXX) -c ../sorting/cpuRaduls.cpp -o cpu.o $(CXXFLAGS) -I.

cpuIntel.o: ../sorting/cpuIntel.cpp
	$(ICPX) -fPIE -c ../sorting/cpuIntel.cpp -o cpuIntel.o

sorting_network.o: ../RADULS/Raduls/sorting_network.cpp
	$(CXX) -c ../RADULS/Raduls/sorting_network.cpp -o sorting_network.o -std=c++17 -O1

clean:
	rm -f $(OBJS) program

run: program
	./program