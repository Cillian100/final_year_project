#include <math.h>
#include <vector>
#include <stdio.h>
#include <random>
#include <algorithm>
#include <numeric>
#include <pthread.h>
#include <fstream>
#include <iostream>
 
extern long long thrust_sorting(std::vector<long long> &data, int device, long long size);
extern long long tbb_sorting(std::vector<long long> &data, long long size);
void printToLatex(std::vector<std::vector<long long>> worstResults, std::vector<std::vector<long long>> bestResults, int steps, std::vector<long long> range);
void printToPython(std::vector<std::vector<long long>> worstResults, std::vector<std::vector<long long>> bestResults, int steps, std::vector<long long> range);
 
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
    std::vector<long long> refCopy = *s->myVector;
    for(int a=0;a<10;a++){
        std::vector<long long> sortData = refCopy;
        s->runtime = s->runtime + thrust_sorting(sortData, 0, s->size);
    }
    s->runtime = s->runtime / 10;
    return NULL;
}
 
void* CPU_sort(void* args){
    CPU_struct *s = (CPU_struct *)args;
    s->runtime = 0;
    std::vector<long long> refCopy = *s->myVector;
    for(int a=0;a<10;a++){
        std::vector<long long> sortData = refCopy;
        s->runtime = s->runtime + tbb_sorting(sortData, s->size);
    }
    s->runtime = s->runtime / 10;
    return NULL;
}
 
std::vector<long long> bestCase(long long length){
    std::vector<long long> returnVectors(2);
    pthread_t thread1;
 
    printf("best case: generating vector (length %lld)...\n", length);
    std::vector<long long> myVector(length);
    std::iota(myVector.begin(), myVector.end(), 0LL);
 
    GPU_struct gpuStruct = {&myVector, 0, length, 0};
    pthread_create(&thread1, NULL, GPU_sort, &gpuStruct);
    pthread_join(thread1, NULL);
    returnVectors.at(0) = gpuStruct.runtime;
    printf("best case GPU done\n");
 
    CPU_struct cpuStruct = {&myVector, length, 0};
    pthread_create(&thread1, NULL, CPU_sort, &cpuStruct);
    pthread_join(thread1, NULL);
    returnVectors.at(1) = cpuStruct.runtime;
    printf("best case CPU done\n");
 
    return returnVectors;
}
 
std::vector<long long> worstCase(long long length){
    std::vector<long long> returnVectors(2);
    pthread_t thread1;
 
    printf("worst case: generating vector (length %lld)...\n", length);
    std::vector<long long> myVector(length);
    std::iota(myVector.rbegin(), myVector.rend(), 0LL);
 
    GPU_struct gpuStruct = {&myVector, 0, length, 0};
    pthread_create(&thread1, NULL, GPU_sort, &gpuStruct);
    pthread_join(thread1, NULL);
    returnVectors.at(0) = gpuStruct.runtime;
    printf("worst case GPU done\n");
 
    CPU_struct cpuStruct = {&myVector, length, 0};
    pthread_create(&thread1, NULL, CPU_sort, &cpuStruct);
    pthread_join(thread1, NULL);
    returnVectors.at(1) = cpuStruct.runtime;
    printf("worst case CPU done\n");
 
    return returnVectors;
}
 
int main(){
    const int steps = 5;
    std::vector<std::vector<long long>> worstResults(steps);
    std::vector<std::vector<long long>> bestResults(steps);
    std::vector<long long> range(steps);
 
    for(int i=0; i<steps; i++){
        long long length = 100000000LL * (i + 1);
        range[i] = 100000000LL * (i + 1);
        worstResults[i] = worstCase(length);
        bestResults[i]  = bestCase(length);
 
        std::cout << "size " << length
                  << " | worst GPU: " << worstResults[i][0] << "ms"
                  << " worst CPU: "   << worstResults[i][1] << "ms"
                  << " | best GPU: "  << bestResults[i][0]  << "ms"
                  << " best CPU: "    << bestResults[i][1]  << "ms"
                  << std::endl;
    }

    printToLatex(worstResults, bestResults, steps, range);
    printToPython(worstResults, bestResults, steps, range);
}
 
void printToLatex(std::vector<std::vector<long long>> worstResults, std::vector<std::vector<long long>> bestResults, int steps, std::vector<long long> range){
    std::ofstream myFile("filename.txt");

    myFile << "\\begin{table}[H]" << std::endl;
    myFile << "\\centering" << std::endl;
    myFile << "\\caption{Comparison of typical \"best case\" and \"worst case\" in GPU and CPU sorting}" << std::endl;
    myFile << "\\vspace{4pt}" << std::endl;
    myFile << "\\begin{tabular}{lcccccc}" << std::endl;
    myFile << "\\toprule" << std::endl;
    myFile << "\\textbf{Input Range} & \\textbf{Unordered GPU} & \\textbf{Ordered GPU} & \\textbf{Unordered CPU} & \\textbf{Ordered CPU} \\\\" << std::endl;
    myFile << "\\midrule" << std::endl;

    for(int a=0;a<steps;a++){
        myFile << range[a] << " & " << worstResults[a][0] << " & " <<  worstResults[a][1] << " & " << bestResults[a][0] << " & " << bestResults[a][1] << "\\\\" << std::endl; 
    }

    myFile << "\\bottomrule" << std::endl;
    myFile << "\\end{tabular}" << std::endl;
    myFile << "\\end{table}" << std::endl;

    myFile.close();
}


void printToPython(std::vector<std::vector<long long>> worstResults, std::vector<std::vector<long long>> bestResults, int steps, std::vector<long long> range){
    std::ofstream myFile("pythonProgram.py");
    myFile << "import matplotlib.pyplot as plt" << std::endl;
    myFile << "import matplotlib.ticker as ticker" << std::endl;
    myFile << "import numpy as np" << std::endl << std::endl;

    myFile << "sizes = [";
    for(int a=0; a<steps; a++){
        myFile << range[a];
        if(a<steps-1){
            myFile << ", ";
        }
    }
    myFile << "]" << std::endl;

    myFile << "worstGPU = [";
    for(int a=0;a<steps; a++){
        myFile << worstResults[a][0];
        if(a<steps-1){
            myFile << ", ";
        }
    }
    myFile << "]" << std::endl;

    myFile << "bestGPU = [";
    for(int a=0;a<steps; a++){
        myFile << bestResults[a][0];
        if(a<steps-1){
            myFile << ", ";
        }
    }
    myFile << "]" << std::endl;

    myFile << "worstCPU = [";
    for(int a=0;a<steps; a++){
        myFile << worstResults[a][1];
        if(a<steps-1){
            myFile << ", ";
        }
    }
    myFile << "]" << std::endl;

    myFile << "bestCPU = [";
    for(int a=0;a<steps; a++){
        myFile << worstResults[a][1];
        if(a<steps-1){
            myFile << ", ";
        }
    }
    myFile << "]" << std::endl;

    myFile << "fig1, ax1 = plt.subplots(figsize=(7, 4.5))" << std::endl;
    myFile << "ax1.plot(sizes, worstGPU, marker='s', linewidth=2, markersize=6, label='worse case GPU')" << std::endl;
    myFile << "ax1.plot(sizes, bestGPU, marker='s', linewidth=2, markersize=6, label='best case GPU')" << std::endl;
    myFile << "ax1.plot(sizes, worstCPU, marker='s', linewidth=2, markersize=6, label='worse case CPU')" << std::endl; 
    myFile << "ax1.plot(sizes, bestCPU, marker='s', linewidth=2, markersize=6, label='best case CPU')" << std::endl;

    myFile << "ax1.set_xlabel('Problem Size (Number of Elements)', fontsize=11)" << std::endl;
    myFile << "ax1.set_ylabel('Runtime (ms)', fontsize=11)" << std::endl;
    myFile << "ax1.set_title('Comparision of sorted vs unsorted array in radix sort', fontsize=13, weight='bold')" << std::endl;
    myFile << "ax1.legend(frameon=True)" << std::endl;
    myFile << "ax1.grid(True, linestyle='--', alpha=0.6)" << std::endl;
    myFile << "ax1.tick_params(axis='both', labelsize=10)" << std::endl;
    myFile << "plt.tight_layout() " << std::endl;
    myFile << "plt.savefigs('../../graphs/bestWorstCase.pdf', bbox_inches='tight') " << std::endl;

    myFile.close();
}
