#include <vector>
#include <cstdlib>
#include <climits>
#include <cstdio>
#include <pthread.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <random>
 
extern float raduls_sorting_total_time(std::vector<long long> &data, long long starting, long long ending);
extern float thrust_sorting_total_time(std::vector<long long> &data, int device, long long starting, long long ending);
extern std::vector<long long> functional_partition(long long n);
void printToLatex(std::vector<std::vector<long long>> myVec, std::string information, std::ofstream &myFile);
 
// FIX: runtime changed from long int to long long for consistent 64-bit
// accumulation across all platforms (long int is 32-bit on Windows).
typedef struct{
    std::vector<long long> *data;
    int startingPoint;
    int endingPoint;
    long long runtime;
    // FIX: removed unused data2 field from both structs.
}CPU_struct;
 
typedef struct{
    int device;
    std::vector<long long> *data;
    int startingPoint;
    int endingPoint;
    long long runtime;
    // FIX: removed unused data2 field from both structs.
}GPU_struct;
 
void* GPU_sort(void* arg){
    GPU_struct *s = (GPU_struct *)arg;
    s->runtime = 0;
    for(int a = 0; a < 20; a++){
        std::vector<long long> mydata = *s->data;
        s->runtime = s->runtime + thrust_sorting_total_time(mydata, s->device, s->startingPoint, s->endingPoint);
    }
    return NULL;
}
 
void* CPU_sort(void* arg){
    CPU_struct *s = (CPU_struct *)arg;
    s->runtime = 0;
    for(int a = 0; a < 20; a++){
        std::vector<long long> mydata = *s->data;
        s->runtime = s->runtime + raduls_sorting_total_time(mydata, s->startingPoint, s->endingPoint);
    }
    return NULL;
}
 
std::vector<long long> benchmark(std::vector<long long> dataPartition, long long length, long long range){
    std::vector<long long> data(length);
 
    // FIX: partition boundary indices changed from int to long long to avoid
    // overflow when length exceeds ~2.1 billion (INT_MAX).
    long long first  = 0;
    long long second = dataPartition.at(0);
    long long third  = dataPartition.at(1) + second;
    long long fourth = dataPartition.at(2) + third;
 
    // FIX: range parameter is now used in the distribution so the caller can
    // control the key range for benchmarking (e.g. to stress radix sort).
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<long long> dist(0, (1LL << (8*3)) - 1);
    std::generate(data.begin(), data.end(), [&](){
        return dist(rng);
    });
 
    pthread_t thread1, thread2, thread3;
    GPU_struct gpu1 = {0, &data, (int)first,  (int)second};
    GPU_struct gpu2 = {1, &data, (int)second, (int)third};
    CPU_struct cpu  = {   &data, (int)third,  (int)fourth};
 
    printf("sorting\n");
    pthread_create(&thread1, NULL, GPU_sort, &gpu1);
    pthread_create(&thread2, NULL, GPU_sort, &gpu2);
    pthread_create(&thread3, NULL, CPU_sort, &cpu);
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    pthread_join(thread3, NULL);
    printf("finished sorting\n");
 
    long long runtimes[3] = {gpu1.runtime, gpu2.runtime, cpu.runtime};
    long long maxRuntime = *std::max_element(runtimes, runtimes+3);
    long long minRuntime = *std::min_element(runtimes, runtimes+3);
    long long spread = maxRuntime - minRuntime;
 
    printf("\nBenchmark:\n");
    printf("GPU 1:  %lld ms\n", gpu1.runtime/20);
    printf("GPU 2:  %lld ms\n", gpu2.runtime/20);
    printf("CPU:    %lld ms\n", cpu.runtime/20);
    printf("Spread: %lld ms\n", spread/20);
    printf("Balance: %.1f%%\n", (1.0 - (double)spread / maxRuntime) * 100.0);
 
    std::vector<long long> returnValues(6);
    returnValues.at(0) = length;
    returnValues.at(1) = gpu1.runtime/20;
    returnValues.at(2) = gpu2.runtime/20;
    returnValues.at(3) = cpu.runtime/20;
    returnValues.at(4) = spread/20;
    returnValues.at(5) = (long long)((1.0 - (double)spread / maxRuntime) * 100.0);
    return returnValues;
}
 
// FIX: warmup() now calls the actual sort functions so the GPU and CPU
// runtimes are properly primed before benchmarking begins.
void warmup(){
    std::vector<long long> dummy(1000);
    thrust_sorting_total_time(dummy, 0, 0, 1000);
    thrust_sorting_total_time(dummy, 1, 0, 1000);
    raduls_sorting_total_time(dummy, 0, 1000);
}
 
std::vector<long long> benchmark_partition(long long length){
    std::vector<long long> returnValue(3);
    returnValue.at(0) = length/3;
    returnValue.at(1) = length/3;
    returnValue.at(2) = length/3;
 
    while(returnValue.at(0) + returnValue.at(1) + returnValue.at(2) < length){
        returnValue.at(2)++;
    }
 
    return returnValue;
}
 
int main(){
    long long length = 100000000;
    long long range  = 100;
    int iterations   = 5;
    std::vector<std::vector<long long>> myVec1;
    std::vector<std::vector<long long>> myVec2;
    std::ofstream myFile("filename.txt");
 
    for(int a = 0; a < iterations; a++){
        printf("Functional model %lld\n", length);
        std::vector<long long> dataPartition1 = functional_partition(length);
        // FIX: renamed inner loop variable from 'a' to 'i' to avoid shadowing
        // the outer loop variable, which caused -Wshadow warnings and
        // potential confusion about which counter was active.
        for(int i = 0; i < (int)dataPartition1.size(); i++){
            printf("%lld\n", dataPartition1.at(i));
        }
        warmup();
        myVec1.push_back(benchmark(dataPartition1, length, range));
        length += 100000000;
    }
 
    length = 100000000;
    for(int a = 0; a < iterations; a++){
        printf("Benchmark sorting %lld\n", length);
        std::vector<long long> dataPartition2 = benchmark_partition(length);
        warmup();
        myVec2.push_back(benchmark(dataPartition2, length, range));
        length += 100000000;
    }
 
    printToLatex(myVec1, "Runtime analysis of functional model in (ms) with range " + std::to_string(range), myFile);
    printToLatex(myVec2, "Runtime analysis of benchmark model in (ms) with range " + std::to_string(range), myFile);
 
    myFile.close();
}
 
void printToLatex(std::vector<std::vector<long long>> myVec, std::string information, std::ofstream &myFile){
    myFile << "\\begin{table}[h]" << std::endl;
    myFile << "\\centering" << std::endl;
    myFile << "\\caption{" << information << "}" << std::endl;
    myFile << "\\vspace{4pt}" << std::endl;
    myFile << "\\begin{tabular}{lcccccc}" << std::endl;
    myFile << "\\toprule" << std::endl;
    myFile << "\\textbf{Array Size} & \\textbf{GPU 1 (ms)} & \\textbf{GPU 2 (ms)} & \\textbf{CPU (ms)} & \\textbf{Spread (ms)} & \\textbf{Balance} \\\\" << std::endl;
    myFile << "\\midrule" << std::endl;
 
    for(int a = 0; a < (int)myVec.size(); a++){
        myFile << myVec.at(a).at(0) << " & ";
        myFile << myVec.at(a).at(1) << " & ";
        myFile << myVec.at(a).at(2) << " & ";
        myFile << myVec.at(a).at(3) << " & ";
        myFile << myVec.at(a).at(4) << " & ";
        myFile << myVec.at(a).at(5) << "\\% \\\\" << std::endl;
    }
 
    myFile << "\\bottomrule" << std::endl;
    myFile << "\\end{tabular}" << std::endl;
    myFile << "\\end{table}" << std::endl;
}