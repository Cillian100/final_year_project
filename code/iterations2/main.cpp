#include <math.h>
#include <vector>
#include <stdio.h>
#include <random>
#include <algorithm>
#include <pthread.h>
#include <fstream>
#include <iostream>

extern long long thrust_sorting(std::vector<long long> &data, int device, long long size);
extern long long tbb_sorting(std::vector<long long> &data, long long size);
void printToLatex(std::vector<long long> gpuRuntime, std::vector<long long> cpuRuntime, int iter);
void printToPython(std::vector<long long> gpuRuntime, std::vector<long long> cpuRuntime, int iter);
void printToPython2(std::vector<float> ratio, int iter);

int variables=4;

typedef struct{
    std::vector<long long> *myVector;
    int device;
    long long size;
    long long runtime;
}GPU_struct;

typedef struct{
    std::vector<long long> *myVector;
    long long size;
    long long runtime;
}CPU_struct;

void* GPU_sort(void* args){
    GPU_struct *s = (GPU_struct *)args;
    s->runtime = 0;
    for(int a=0;a<variables;a++){
        std::vector<long long> myData = *s->myVector;
        s->runtime = s->runtime + thrust_sorting(myData, 0, s->size);
    }
    return NULL;
}

void* CPU_sort(void* args){
    CPU_struct *s = (CPU_struct *)args;
    s->runtime = 0;
    for(int a=0;a<variables;a++){
        std::vector<long long> myData = *s->myVector;
        s->runtime = s->runtime + tbb_sorting(myData, s->size);
    }
    return NULL;
}

long long iterations(int multiplier, long long length){
    pthread_t thread1;

    std::mt19937_64 rng(std::random_device{}());

    std::vector<long long> myVector(length);

    if(multiplier >= 8){
        std::uniform_int_distribution<unsigned long long> dist(0, UINT64_MAX);
        std::generate(myVector.begin(), myVector.end(), [&](){
            return (long long)dist(rng);
        });
        //printf("UINT64_MAX ");
    } else {
        std::uniform_int_distribution<long long> dist(0, (1LL << (8*multiplier)) - 1);
        std::generate(myVector.begin(), myVector.end(), [&](){
            return dist(rng);
        });
        //printf("%lld ", 1LL << (8*multiplier));
    }

    GPU_struct myStruct = {&myVector, 0, length, 0};
    pthread_create(&thread1, NULL, GPU_sort, &myStruct);
    pthread_join(thread1, NULL);

    //std::cout << "GPU runtime " << myStruct.runtime/variables << std::endl;
    return (myStruct.runtime/variables);
}

long long iterations2(int multiplier, long long length){
    pthread_t thread1;

    std::mt19937_64 rng(std::random_device{}());

    std::vector<long long> myVector(length);

    if(multiplier >= 8){
        std::uniform_int_distribution<unsigned long long> dist(0, UINT64_MAX);
        std::generate(myVector.begin(), myVector.end(), [&](){
            return (long long)dist(rng);
        });
    } else {
        std::uniform_int_distribution<long long> dist(0, (1LL << (8*multiplier)) - 1);
        std::generate(myVector.begin(), myVector.end(), [&](){
            return dist(rng);
        });
    }

    CPU_struct myStruct = {&myVector, length, 0};
    pthread_create(&thread1, NULL, CPU_sort, &myStruct);
    pthread_join(thread1, NULL);

    //printf("runtime %lld\n", myStruct.runtime/10);
    //std::cout << "CPU runtime " << myStruct.runtime/variables << std::endl;
    return (myStruct.runtime/variables);    
}

void function(long long length){
    int iter=8;
    std::vector<long long> gpuRuntime(iter);
    std::vector<long long> cpuRuntime(iter);
    std::vector<float> ratio(iter);

    for(int a=0;a<iter;a++){
        gpuRuntime.at(a)=iterations(a+1, length);
        cpuRuntime.at(a)=iterations2(a+1, length);
    }

    for(int a=0;a<iter;a++){
        ratio.at(a)=(float)gpuRuntime.at(a)/(float)cpuRuntime.at(a);
    }

    printf("Length: %lld\n", length);
    printf("GPU runtime: ");
    for(int a=0;a<iter;a++){
        printf("%lld ", gpuRuntime.at(a));
    }
    printf("\n");
    printf("CPU runtime: ");
    for(int a=0;a<iter;a++){
        printf("%lld ", cpuRuntime.at(a));
    }
    printf("\n");
    printf("Ratio: ");
    for(int a=0;a<iter;a++){
        printf("%f ", ratio.at(a));
    }
    printf("\n\n");
}

int main(){
    int length=10000000;
    for(int a=0;a<10;a++){
        function(length);
        length=length*2;
    }
}

void printToLatex(std::vector<long long> gpuRuntime, std::vector<long long> cpuRuntime, int iterations){
    std::ofstream myFile("filename_10000000.txt");

    myFile << "\\begin{table}[H]" << std::endl;
    myFile << "\\centering" << std::endl;
    myFile << "\\caption{Runtime Analysis of Functional Model in (ms) with Range}" << std::endl;
    myFile << "\\vspace{4pt}" << std::endl;
    myFile << "\\begin{tabular}{lcccccc}" << std::endl;
    myFile << "\\toprule" << std::endl;
    myFile << "\\textbf{Input range} & \\textbf{GPU (ms)} & \\textbf{CPU (ms)} & \\textbf{ratio} \\\\" << std::endl;
    myFile << "\\midrule" << std::endl;

    for(int a=0;a<iterations;a++){
        myFile << "$2^{8*" << (a+1) << "}$ & " << gpuRuntime.at(a) << " & " << cpuRuntime.at(a) << " & " << ((float)gpuRuntime.at(a)/(float)cpuRuntime.at(a)) << " \\\\" << std::endl; 
    }
    

    myFile << "\\bottomrule" << std::endl;
    myFile << "\\end{tabular}" << std::endl;
    myFile << "\\end{table}" << std::endl;

    myFile.close();
}

void printToPython2(std::vector<float> ratio, int iterations){
    std::ofstream myFile("pythonProgram2_10000000.py");

    myFile << "import matplotlib.pyplot as plt" << std::endl;
    myFile << "import matplotlib.ticker as ticker" << std::endl;
    myFile << "import numpy as np" << std::endl << std::endl;

    myFile << "sizes = [";
    for(int a=1; a<=iterations; a++){
        if(a < 8){
            myFile << (1ULL << (8*a));
        } else {
            myFile << "18446744073709551616";
        }
        if(a < iterations){
            myFile << ", ";
        }
    }
    myFile << "]" << std::endl;
    
    myFile << "ratio = [";
    for(int a=0;a<iterations;a++){
        myFile << ratio[a];
        if(a<iterations-1){
            myFile << ", ";
        }
    }
    myFile << "]" << std::endl;

    myFile << "fig1, ax1 = plt.subplots(figsize=(7, 4.5))" << std::endl;
    myFile << "ax1.plot(sizes, ratio, marker='s', linewidth=2, markersize=6, label='ratio of CPU MSD vs GPU LSD')" << std::endl;
    myFile << "ax1.set_xscale('log')" << std::endl;
    myFile << "ax1.set_xlabel('Range of Input Elements', fontsize=11)" << std::endl;
    myFile << "ax1.set_ylabel('Algorithm Runtime in MS', fontsize=11)" << std::endl;
    myFile << "ax1.set_title('MSD vs LSD Radix Sort ratio', fontsize=11)" << std::endl;
    myFile << "ax1.legend(frameon=True)" << std::endl;
    myFile << "ax1.grid(True, linestyle='--', alpha=0.6)" << std::endl;
    myFile << "ax1.tick_params(axis='both', labelsize=10)" << std::endl;
    myFile << "plt.tight_layout()" << std::endl;
    myFile << "plt.savefig('../../graphs/iterations2.pdf', bbox_inches='tight')" << std::endl;

    myFile << std::endl;
    myFile.close();
}

void printToPython(std::vector<long long> gpuRuntime, std::vector<long long> cpuRuntime, int iterations){
    std::ofstream myFile("pythonProgram_10000000.py");

    myFile << "import matplotlib.pyplot as plt" << std::endl;
    myFile << "import matplotlib.ticker as ticker" << std::endl;
    myFile << "import numpy as np" << std::endl << std::endl;

    myFile << "sizes = [";
    for(int a=1; a<=iterations; a++){
        if(a < 8){
            myFile << (1ULL << (8*a));
        } else {
            myFile << "18446744073709551616";
        }
        if(a < iterations){
            myFile << ", ";
        }
    }
    myFile << "]" << std::endl;

    myFile << "gpuRuntime = [";
    for(int a=0;a<iterations;a++){
        myFile << gpuRuntime[a];
        if(a<iterations-1){
            myFile << ", ";
        }
    }
    myFile << "]" << std::endl;

    myFile << "cpuRuntime = [";
    for(int a=0;a<iterations;a++){
        myFile << cpuRuntime[a];
        if(a<iterations-1){
            myFile << ", ";
        }
    }
    myFile << "]" << std::endl;

    myFile << "fig1, ax1 = plt.subplots(figsize=(7, 4.5))" << std::endl;
    myFile << "ax1.plot(sizes, cpuRuntime, marker='s', linewidth=2, markersize=6, label='CPU MSD Radix Sort')" << std::endl;
    myFile << "ax1.plot(sizes, gpuRuntime, marker='s', linewidth=2, markersize=6, label='GPU LSD Radix Sort') "<< std::endl;
    myFile << "ax1.set_xscale('log')" << std::endl;
    myFile << "ax1.set_xlabel('Range of Input Elements', fontsize=11)" << std::endl;
    myFile << "ax1.set_ylabel('Algorithm Runtime in MS', fontsize=11)" << std::endl;
    myFile << "ax1.set_title('MSD vs LSD Radix Sort', fontsize=11)" << std::endl;
    myFile << "ax1.legend(frameon=True)" << std::endl;
    myFile << "ax1.grid(True, linestyle='--', alpha=0.6)" << std::endl;
    myFile << "ax1.tick_params(axis='both', labelsize=10)" << std::endl;
    myFile << "plt.tight_layout()" << std::endl;
    myFile << "plt.savefig('../../graphs/iterations1.pdf', bbox_inches='tight')" << std::endl;

    myFile << std::endl;
    myFile.close();

}