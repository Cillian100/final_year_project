#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <chrono>

double measuring_cuda_speed(int n, int repeats, std::vector<long long> &data){
    //thrust::device_vector<int> data(n);
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
    double seconds_per_sort = (ms / 1000.0) / repeats;
    double melems_per_sec = (n / seconds_per_sort) / 1e6;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("%d %f\n", n, melems_per_sec);
    return melems_per_sec;
}

//int main(){
//    int n=1000, repeats=5, steps=18;
//    std::vector<double> speeds(steps);

//    for(int a=0;a<200000000;a++){
//        data[a]=rand()%10000000;
//    }

//    for(int a=1; a<=steps; a++){
//        speeds[a-1] = measuring_cuda_speed(n*a, repeats, data);
//        n=n*2;
//    }

//    printf("sizes = [");
//    for(int a=1; a<=steps; a++){
        //printf("%d, ", n*a);
//        printf("%d%s", n*a, a<steps ? ", " : "");

//    }
//    printf("]\n");

//    printf("thrust = [");
//    for(int a=0; a<steps; a++){
//        printf("%.0f%s", speeds[a], a<steps-1 ? ", " : "");
//    }
//    printf("]\n");

//    return 0;
//}