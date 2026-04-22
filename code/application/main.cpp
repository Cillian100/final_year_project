#include <vector>
#include <cstdlib>
#include <climits>
#include <cstdio>
#include <pthread.h>
#include <algorithm>
#include <iostream>
#include <fstream>

extern long int tbb_sorting(int starting, int ending, std::vector<long long> &data);
extern long int thrust_sorting(int starting, int ending, int device, std::vector<long long> &data);
extern std::vector<long long> functional_partition(long long n);
void printToLatex(std::vector<std::vector<long long>> myVec, std::string information, std::ofstream &myFile);

typedef struct{
    std::vector<long long> *data;
    std::vector<long long> *data2;
    int startingPoint;
    int endingPoint;
    long int runtime;
}CPU_struct;


typedef struct{
    int device;
    std::vector<long long> *data;
    std::vector<long long> *data2;
    int startingPoint;
    int endingPoint;
    long int runtime;
}GPU_struct;

void* GPU_sort(void* arg){
    GPU_struct *s = (GPU_struct *)arg;
    s->runtime = 0;
    for(int a = 0; a < 20; a++){
        std::vector<long long> mydata = *s->data;
        s->runtime = s->runtime + thrust_sorting(s->startingPoint, s->endingPoint, s->device, mydata);
    }
    return NULL;
}

void* CPU_sort(void* arg){
    CPU_struct *s = (CPU_struct *)arg;
    s->runtime=0;
    for(int a=0;a<20;a++){
        std::vector<long long> mydata = *s->data;
        s->runtime = s->runtime + tbb_sorting(s->startingPoint, s->endingPoint, mydata);
    }
    return NULL;
}

std::vector<long long> benchmark(std::vector<long long> dataPartition, int length, long long range){
    std::vector<long long> data(length);
    int first=0;
    int second=dataPartition.at(0);
    int third=dataPartition.at(1)+second;
    int fourth=dataPartition.at(2)+third;
    long int totaldif;

    //printf("generating dataset\n");
    for(long long a=0;a<length;a++){
        data[a]=rand()%range;
    }

    pthread_t thread1, thread2, thread3;
    GPU_struct gpu1 = {0, &data, NULL, first, second};
    GPU_struct gpu2 = {1, &data, NULL, second, third};
    CPU_struct cpu = {&data, NULL, third, fourth};

    //printf("sorting\n");
    pthread_create(&thread1, NULL, GPU_sort, &gpu1);
    pthread_create(&thread2, NULL, GPU_sort, &gpu2);
    pthread_create(&thread3, NULL, CPU_sort, &cpu);
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    pthread_join(thread3, NULL);
    //printf("finished sorting\n");

    long int runtimes[3] = {gpu1.runtime, gpu2.runtime, cpu.runtime};
    long int maxRuntime = *std::max_element(runtimes, runtimes+3);
    long int minRuntime = *std::min_element(runtimes, runtimes+3);
    long int spread = maxRuntime - minRuntime;

    //printf("\nBenchmark:\n");
    //printf("GPU 1:  %ld ms\n", gpu1.runtime/20);
    //printf("GPU 2:  %ld ms\n", gpu2.runtime/20);
    //printf("CPU:    %ld ms\n", cpu.runtime/20);
    //printf("Spread: %ld ms\n", spread/20);
    //printf("Balance: %.1f%%\n", (1.0 - (double)spread / maxRuntime) * 100.0);
    std::vector<long long> returnValues(6);
    returnValues.at(0)=length;
    returnValues.at(1)=gpu1.runtime/20;
    returnValues.at(2)=gpu2.runtime/20;
    returnValues.at(3)=cpu.runtime/20;
    returnValues.at(4)=spread/20;
    returnValues.at(5)=((1.0 - (double)spread / maxRuntime)*100.0);
    return returnValues;
}

void warmup(){
    std::vector<long long> dummy(1000);
    thrust_sorting(0, 1000, 0, dummy);
    thrust_sorting(0, 1000, 1, dummy);
    tbb_sorting(0, 1000, dummy);
}

std::vector<long long> benchmark_partition(long long length){
    std::vector<long long> returnValue(3);
    returnValue.at(0)=length/3;
    returnValue.at(1)=length/3;
    returnValue.at(2)=length/3;

    while(returnValue.at(0) + returnValue.at(1) + returnValue.at(2) < length){
        returnValue.at(2)++;
    }

    return returnValue;
}

int main(){
    long long length=100000000;
    long long range=100;
    int iterations=5;
    std::vector<std::vector<long long>> myVec1;
    std::vector<std::vector<long long>> myVec2;
    std::ofstream myFile("filename.txt");

    for(int a=0;a<iterations;a++){
        printf("Functional model %lld\n", length);
        std::vector<long long> dataPartition1=functional_partition(length);
        warmup();
        myVec1.push_back(benchmark(dataPartition1, length, range));
        length=length+100000000;
    }

    length=100000000;
    for(int a=0;a<iterations;a++){
        printf("Benchmark sorting %lld\n", length);
        std::vector<long long> dataPartition2=benchmark_partition(length);
        warmup();
        myVec2.push_back(benchmark(dataPartition2, length, range));
        length=length+100000000;
    }

    printToLatex(myVec1, "Runtime analysis of functional model in (ms) with range " + range, myFile);
    printToLatex(myVec2, "Runtime analysis of benchmark model in (ms) with range " + range, myFile);

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

    for(int a=0;a<myVec.size();a++){
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