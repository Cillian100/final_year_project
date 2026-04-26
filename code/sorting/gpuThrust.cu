#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <time.h>

float thrust_sorting(std::vector<long long> &data, int device, long long starting, long long ending){
    cudaSetDevice(device);
    thrust::device_vector<long long> buf(ending-starting);
    auto begin = std::chrono::high_resolution_clock::now();
    
    thrust::copy(data.begin()+starting, data.begin()+ending, buf.begin());
    thrust::sort(buf.begin(), buf.end());
    thrust::copy(buf.begin(), buf.begin(), data.begin()+starting);
    
    auto end = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<double, std::milli>(end - begin).count();
    return (ending-starting)/ms;
}

float thrust_sorting_total_time(std::vector<long long> &data, int device, long long starting, long long ending){
    cudaSetDevice(device);
    thrust::device_vector<long long> buf(ending-starting);
    auto begin = std::chrono::high_resolution_clock::now();
    
    thrust::copy(data.begin()+starting, data.begin()+ending, buf.begin());
    thrust::sort(buf.begin(), buf.end());
    thrust::copy(buf.begin(), buf.begin(), data.begin()+starting);
    
    auto end = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<double, std::milli>(end - begin).count();
    return ms;
}

long long thrust_sorting_start(int starting, int ending, std::vector<long long> &data, int device){
    struct timespec t0, t1;
    long long size = ending - starting;
    cudaSetDevice(device);
    clock_gettime(CLOCK_MONOTONIC, &t0);

    thrust::device_vector<long long> buf(size);
    thrust::copy(data.begin()+starting, data.begin()+ending, buf.begin());
    thrust::sort(buf.begin(), buf.end());
    thrust::copy(buf.begin(), buf.end(), data.begin()+starting);

    clock_gettime(CLOCK_MONOTONIC, &t1);

    return (long long)((t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000);
}

double measuring_thrust_speed(std::vector<long long> &data, int device, long long n){
    cudaSetDevice(device);
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
    //printf("%lld %f\n", n, melems_per_sec);
    return melems_per_sec;
}