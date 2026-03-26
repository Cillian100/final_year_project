#include <vector>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

double measuring_tbb_speed(int n, std::vector<long long> &data){
    std::vector<long long> buf(n);
    auto start = std::chrono::high_resolution_clock::now();

    std::copy(data.begin(), data.begin()+n, buf.begin());
    oneapi::dpl::sort(oneapi::dpl::execution::par_unseq, buf.begin(), buf.begin()+n);

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double seconds_per_sort = (ms / 1000);
    double melems_per_sec = (n / seconds_per_sort) / 1e6;

    printf("%d %f\n", n, melems_per_sec);

    return melems_per_sec;
}