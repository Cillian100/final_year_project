#include <vector>
#include <thread>
#include <time.h>
#include <cstring>
#include "../RADULS/Raduls/raduls.h"
 
long long tbb_sorting(std::vector<long long> &data, long long size){
    struct timespec t0, t1;
 
    uint32_t n_threads = std::thread::hardware_concurrency();
    uint64_t bytes = size * sizeof(long long);
    
    auto* _raw_input = new uint8_t[bytes + raduls::ALIGNMENT];
    auto* _raw_tmp   = new uint8_t[bytes + raduls::ALIGNMENT];
 
    auto* input = _raw_input;
    auto* tmp   = _raw_tmp;
 
    while(reinterpret_cast<uintptr_t>(input) % raduls::ALIGNMENT) ++input;
    while(reinterpret_cast<uintptr_t>(tmp)   % raduls::ALIGNMENT) ++tmp;
 
    memcpy(input, data.data(), bytes);
 
    raduls::CleanTmpArray(tmp, size, sizeof(long long), n_threads);
 
    clock_gettime(CLOCK_MONOTONIC, &t0);
    raduls::RadixSortMSD(input, tmp, size, sizeof(long long), sizeof(long long), n_threads);
    clock_gettime(CLOCK_MONOTONIC, &t1);
 
    memcpy(data.data(), input, bytes);
 
    delete[] _raw_input;
    delete[] _raw_tmp;
 
    return (long long)((t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000);
}