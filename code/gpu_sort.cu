#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

void hello_from_cuda(){
	printf("Hello from Cuda\n");
}

void cuda_thrust_sort(std::vector<long int> &myVector, int deviceNumber, long int startingPoint, long int endingPoint){
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if(deviceNumber>deviceCount){
		printf("Insufficient GPUs\n");
		return;
	}	
	cudaSetDevice(deviceNumber);
	
	thrust::device_vector<long int> deviceVector(myVector.begin()+startingPoint, myVector.begin()+endingPoint);
	thrust::sort(deviceVector.begin(), deviceVector.end());
	thrust::copy(deviceVector.begin(), deviceVector.end(), myVector.begin()+startingPoint);
}
