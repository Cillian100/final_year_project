#include <vector>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <chrono>
#include <cstring>

long long tbb_sorting(std::vector<long long> &data, long long size){
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    oneapi::dpl::sort(oneapi::dpl::execution::par_unseq, data.begin(), data.end());
    
    clock_gettime(CLOCK_MONOTONIC, &t1);

    return (long long)((t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000);
}