#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <time.h>

long int thrust_sorting(int starting, int ending, int device, std::vector<long long> &data){
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    
    cudaSetDevice(device);
    thrust::device_vector<long long> buf(ending-starting);
    thrust::copy(data.begin()+starting, data.begin()+ending, buf.begin());
    thrust::sort(buf.begin(), buf.begin()+ending-starting);
    thrust::copy(buf.begin(), buf.begin()+ending-starting, data.begin()+starting);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000;
}