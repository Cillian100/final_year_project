#include <tbb/parallel_sort.h>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <vector>
#include <iostream>
#include <chrono>

extern double measuring_cuda_speed(int n, int repeats, std::vector<long long> &data);


double measuring_tbb_speed(int n, int repeats, std::vector<long long> &data){
    std::vector<long long> buf(n);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::copy(data.begin(), data.begin()+n, buf.begin());
    oneapi::dpl::sort(buf.begin(), buf.begin()+n);
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double seconds_per_sort = (ms / 1000) / repeats;

    double melems_per_sec = (n / seconds_per_sort) / 1e6;

    printf("%d %f\n", n, melems_per_sec);
    return melems_per_sec;
}

int main(){
    int n=1000, repeats=5, steps=16;
    std::vector<double> speeds1(steps);
    std::vector<double> speeds2(steps);
    std::vector<long long> data(1000000000);

    //std::uniform_int_distribution<int> dist(0, 10);
    //for(int a=0;a<10;a++){
    //    printf("%d ", dist(a));
    //}

    for(int a=0;a<1000000000;a++){
        data[a]=rand()%LLONG_MAX;
    }

    for(int a=1; a<=steps; a++){
        printf("%d ", a);
        speeds1[a-1] = measuring_tbb_speed(n*a, repeats, data);
        n=n*2;
    }

    n=1000;
    for(int a=1; a<=steps; a++){
        printf("%d ", a);
        speeds2[a-1] = measuring_cuda_speed(n*a, repeats, data);
        n=n*2;
    }

    n=1000;
    printf("sizes = [");
    for(int a=1; a<=steps; a++){
        printf("%d%s", n, a<steps ? ", " : "");
        n=n*2;

    }
    printf("]\n");

    printf("tbb = [");
    for(int a=0; a<steps; a++){
        printf("%.0f%s", speeds1[a], a<steps-1 ? ", " : "");
    }
    printf("]\n\n\n");

    n=1000;
    printf("sizes = [");
    for(int a=1; a<=steps; a++){
        printf("%d%s", n, a<steps ? ", " : "");
        n=n*2;
    }
    printf("]\n");

    printf("thrust = [");
    for(int a=0; a<steps; a++){
        printf("%.0f%s", speeds2[a], a<steps-1 ? ", " : "");
    }
    printf("]\n");

    return 0;
}