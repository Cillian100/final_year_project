#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>

double measuring_cuda_speed(int n, std::vector<long long> &data){
    thrust::device_vector<long long> buf(n);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);

    thrust::copy(data.begin(), data.begin() + n, buf.begin());
    thrust::sort(buf.begin(), buf.begin() + n);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double seconds_per_sort = (ms / 1000.0);
    double melems_per_sec = (n / seconds_per_sort) / 1e6;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("%d %f\n", n, melems_per_sec);
    return melems_per_sec;
}