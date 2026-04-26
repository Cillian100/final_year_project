#include <vector>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <chrono>

float tbb_sorting(std::vector<long long> &data, long long starting, long long ending){
    std::vector<long long> buf(ending-starting);
    auto start = std::chrono::high_resolution_clock::now();

    std::copy(data.begin()+starting, data.begin()+ending, buf.begin());
    oneapi::dpl::sort(oneapi::dpl::execution::par_unseq, buf.begin(), buf.end());

    auto end = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(end - start).count();

    return (ending-starting)/ms;
}