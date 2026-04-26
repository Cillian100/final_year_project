#include <vector>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <chrono>

long int tbb_sorting(int starting, int ending, std::vector<long long> &data){
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    oneapi::dpl::sort(oneapi::dpl::execution::par_unseq, data.begin()+starting, data.begin()+ending);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000;
}