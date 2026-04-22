#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <time.h>

long long thrust_sorting(std::vector<long long> &data, int device, long long size){
    struct timespec t0, t1;
    cudaSetDevice(device);
    clock_gettime(CLOCK_MONOTONIC, &t0);

    thrust::device_vector<long long> buf(size);
    thrust::copy(data.begin(), data.begin()+size, buf.begin());
    thrust::sort(buf.begin(), buf.begin()+size);
    thrust::copy(buf.begin(), buf.begin()+size, data.begin());

    clock_gettime(CLOCK_MONOTONIC, &t1);

    return (long long)((t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000);
}