#include <vector>
#include <cstdlib>
#include <climits>
#include <cstdio>
#include <pthread.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <random>
#include <malloc.h>


extern float raduls_sorting(std::vector<long long> &data, long long starting, long long ending);
extern float tbb_sorting(std::vector<long long> &data, long long starting, long long ending);


int main(){
    long long size=10000;
    int iterations=18;
    std::vector<long long> mySize(iterations);
    std::vector<float> cpuTime1(iterations);
    std::vector<float> cpuTime2(iterations);
    
    for(int a=0;a<iterations;a++){
        mySize[a]=size;
        std::vector<long long> data(size);
        std::mt19937_64 rng(std::random_device{}());
        printf("%d generating..\n", a);
        std::uniform_int_distribution<long long> dist(0, (1LL << (8*2)) - 1);
        std::generate(data.begin(), data.end(), [&](){
            return dist(rng);
        });
        
        printf("%d sorting..\n", a);
        cpuTime1[a] = raduls_sorting(data, 0, size);
        size=size*2;
    }

    malloc_trim(0);
    size=10000;

    for(int a=0;a<iterations;a++){
        std::vector<long long> data(size);
        std::mt19937_64 rng(std::random_device{}());
        printf("%d generating..\n", a);
        std::uniform_int_distribution<long long> dist(0, (1LL << (8*2)) - 1);
        std::generate(data.begin(), data.end(), [&](){
            return dist(rng);
        });
        
        printf("%d sorting..\n", a);
        cpuTime2[a] = tbb_sorting(data, 0, size);
        size=size*2;
    }

    for(int a=0;a<iterations;a++){
        std::cout << mySize[a] << " ";
    }
    std::cout << std::endl;
    
    for(int a=0;a<iterations;a++){
        std::cout << cpuTime1[a] << " ";
    }
    std::cout << std::endl;

    for(int a=0;a<iterations;a++){
        std::cout << cpuTime2[a] << " ";
    }
    std::cout << std::endl;
    
}