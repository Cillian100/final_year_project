#include <vector>
#include <thread>
#include <time.h>
#include <cstring>
#include <chrono>
#include "../RADULS/Raduls/raduls.h"

void parallel_memcpy(void* dst, const void* src, size_t bytes, uint32_t n_threads){
    uint8_t* d = static_cast<uint8_t*>(dst);
    const uint8_t* s = static_cast<const uint8_t*>(src);

    size_t chunk = bytes / n_threads;
    std::vector<std::thread> threads;

    for(uint32_t t = 0; t < n_threads; t++){
        size_t offset = t * chunk;
        size_t size   = (t == n_threads - 1) ? (bytes - offset) : chunk;
        threads.emplace_back([=](){
            memcpy(d + offset, s + offset, size);
        });
    }
    for(auto& t : threads) t.join();
}

float raduls_sorting(std::vector<long long> &data, long long starting, long long ending){
    long long size = ending - starting;
    uint32_t n_threads = std::thread::hardware_concurrency();
    uint64_t bytes = size * sizeof(long long);

    auto* _raw_input = new uint8_t[bytes + raduls::ALIGNMENT];
    auto* _raw_tmp   = new uint8_t[bytes + raduls::ALIGNMENT];

    auto* input = _raw_input;
    auto* tmp   = _raw_tmp;

    while(reinterpret_cast<uintptr_t>(input) % raduls::ALIGNMENT) ++input;
    while(reinterpret_cast<uintptr_t>(tmp)   % raduls::ALIGNMENT) ++tmp;

    auto begin = std::chrono::high_resolution_clock::now();
    parallel_memcpy(input, data.data() + starting, bytes, 64);

    raduls::CleanTmpArray(tmp, size, sizeof(long long), n_threads);
    raduls::RadixSortMSD(input, tmp, size, sizeof(long long), sizeof(long long), n_threads);

    auto end = std::chrono::high_resolution_clock::now();
    parallel_memcpy(data.data() + starting, input, bytes, 64);


    delete[] _raw_input;
    delete[] _raw_tmp;

    float ms = std::chrono::duration<double, std::milli>(end - begin).count();
    return (ending-starting)/ms;
}

float raduls_sorting_total_time(std::vector<long long> &data, long long starting, long long ending){
    long long size = ending - starting;
    uint32_t n_threads = std::thread::hardware_concurrency();
    uint64_t bytes = size * sizeof(long long);

    auto* _raw_input = new uint8_t[bytes + raduls::ALIGNMENT];
    auto* _raw_tmp   = new uint8_t[bytes + raduls::ALIGNMENT];

    auto* input = _raw_input;
    auto* tmp   = _raw_tmp;

    while(reinterpret_cast<uintptr_t>(input) % raduls::ALIGNMENT) ++input;
    while(reinterpret_cast<uintptr_t>(tmp)   % raduls::ALIGNMENT) ++tmp;

    auto begin = std::chrono::high_resolution_clock::now();
    parallel_memcpy(input, data.data() + starting, bytes, 64);

    raduls::CleanTmpArray(tmp, size, sizeof(long long), n_threads);
    raduls::RadixSortMSD(input, tmp, size, sizeof(long long), sizeof(long long), n_threads);

    auto end = std::chrono::high_resolution_clock::now();
    parallel_memcpy(data.data() + starting, input, bytes, 64);


    delete[] _raw_input;
    delete[] _raw_tmp;

    float ms = std::chrono::duration<double, std::milli>(end - begin).count();
    return ms;
}

long long raduls_sorting_start(int starting, int ending, std::vector<long long> &data){
    struct timespec t0, t1;

    long long size = ending - starting;
    uint32_t n_threads = std::thread::hardware_concurrency();
    uint64_t bytes = size * sizeof(long long);

    auto* _raw_input = new uint8_t[bytes + raduls::ALIGNMENT];
    auto* _raw_tmp   = new uint8_t[bytes + raduls::ALIGNMENT];

    auto* input = _raw_input;
    auto* tmp   = _raw_tmp;

    while(reinterpret_cast<uintptr_t>(input) % raduls::ALIGNMENT) ++input;
    while(reinterpret_cast<uintptr_t>(tmp)   % raduls::ALIGNMENT) ++tmp;

    parallel_memcpy(input, data.data() + starting, bytes, 64);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    raduls::CleanTmpArray(tmp, size, sizeof(long long), n_threads);
    raduls::RadixSortMSD(input, tmp, size, sizeof(long long), sizeof(long long), n_threads);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    parallel_memcpy(data.data() + starting, input, bytes, 64);

    delete[] _raw_input;
    delete[] _raw_tmp;

    return (long long)((t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000);
}

double measuring_raduls_speed(std::vector<long long> &data, long long n){
    uint32_t n_threads = std::thread::hardware_concurrency();
    uint64_t bytes = n * sizeof(long long);
 
    auto* _raw_input = new uint8_t[bytes + raduls::ALIGNMENT];
    auto* _raw_tmp   = new uint8_t[bytes + raduls::ALIGNMENT];
 
    auto* input = _raw_input;
    auto* tmp   = _raw_tmp;
 
    while(reinterpret_cast<uintptr_t>(input) % raduls::ALIGNMENT) ++input;
    while(reinterpret_cast<uintptr_t>(tmp)   % raduls::ALIGNMENT) ++tmp;
 
    auto start = std::chrono::high_resolution_clock::now();
    parallel_memcpy(input, data.data(), bytes, 64);
 
    //auto start = std::chrono::high_resolution_clock::now();
    raduls::CleanTmpArray(tmp, n, sizeof(long long), n_threads);
    raduls::RadixSortMSD(input, tmp, n, sizeof(long long), sizeof(long long), n_threads);
    //auto end = std::chrono::high_resolution_clock::now();
 
    long long* sorted = reinterpret_cast<long long*>(input);
    parallel_memcpy(data.data(), input, bytes, 64);

    auto end = std::chrono::high_resolution_clock::now();
 
    delete[] _raw_input;
    delete[] _raw_tmp;
 
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double melems_per_sec = (n / (ms / 1000.0)) / 1e6;
 
    //printf("%lld %f\n", n, melems_per_sec);
    return melems_per_sec;
}
