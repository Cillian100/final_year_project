#include <math.h>
#include <vector>
#include <stdio.h>
#include <random>
#include <algorithm>
#include <pthread.h>
#include <fstream>

extern long long thrust_sorting(std::vector<long long> &data, int device, long long size);
extern long long tbb_sorting(std::vector<long long> &data, long long size);
void printToLatex(std::vector<long long> gpuRuntime, std::vector<long long> cpuRuntime);

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
    for(int a=0;a<10;a++){
        std::vector<long long> myData = *s->myVector;
        s->runtime = s->runtime + thrust_sorting(myData, 0, s->size);
    }
    return NULL;
}

void* CPU_sort(void* args){
    CPU_struct *s = (CPU_struct *)args;
    s->runtime = 0;
    for(int a=0;a<10;a++){
        std::vector<long long> myData = *s->myVector;
        s->runtime = s->runtime + tbb_sorting(myData, s->size);
    }
    return NULL;
}

long long iterations(int multiplier){
    long long length = 100000000;
    pthread_t thread1;

    std::mt19937_64 rng(std::random_device{}());

    std::vector<long long> myVector(length);

    if(multiplier >= 8){
        std::uniform_int_distribution<unsigned long long> dist(0, UINT64_MAX);
        std::generate(myVector.begin(), myVector.end(), [&](){
            return (long long)dist(rng);
        });
        printf("UINT64_MAX ");
    } else {
        std::uniform_int_distribution<long long> dist(0, (1LL << (8*multiplier)) - 1);
        std::generate(myVector.begin(), myVector.end(), [&](){
            return dist(rng);
        });
        printf("%lld ", 1LL << (8*multiplier));
    }

    GPU_struct myStruct = {&myVector, 0, length, 0};
    pthread_create(&thread1, NULL, GPU_sort, &myStruct);
    pthread_join(thread1, NULL);

    //printf("runtime %lld\n", myStruct.runtime/10);
    return (myStruct.runtime/10);
}

long long iterations2(int multiplier){
    long long length = 100000000;
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

    CPU_struct myStruct = {&myVector, length, 0};
    pthread_create(&thread1, NULL, CPU_sort, &myStruct);
    pthread_join(thread1, NULL);

    //printf("runtime %lld\n", myStruct.runtime/10);
    return (myStruct.runtime/10);    
}


int main(){
    std::vector<long long> gpuRuntime(8);
    std::vector<long long> cpuRuntime(8);
    for(int a=0;a<8;a++){
        gpuRuntime.at(a)=iterations(a+1);
        cpuRuntime.at(a)=iterations2(a+1);
    }

    printToLatex(gpuRuntime, cpuRuntime);
}

void printToLatex(std::vector<long long> gpuRuntime, std::vector<long long> cpuRuntime){
    std::ofstream myFile("filename.txt");

    myFile << "\\begin{table}[H]" << std::endl;
    myFile << "\\centering" << std::endl;
    myFile << "\\caption{Runtime Analysis of Functional Model in (ms) with Range}" << std::endl;
    myFile << "\\vspace{4pt}" << std::endl;
    myFile << "\\begin{tabular}{lcccccc}" << std::endl;
    myFile << "\\toprule" << std::endl;
    myFile << "\\textbf{Input range} & \\textbf{GPU (ms)} & \\textbf{CPU (ms)} & \\textbf{ratio} \\\\" << std::endl;
    myFile << "\\midrule" << std::endl;

    for(int a=0;a<8;a++){
        myFile << "$2^{8*" << a << "}$ & " << gpuRuntime.at(a) << " & " << cpuRuntime.at(a) << " & " << ((float)gpuRuntime.at(a)/(float)cpuRuntime.at(a)) << " \\\\" << std::endl; 
    }

    myFile << "\\bottomrule" << std::endl;
    myFile << "\\end{tabular}" << std::endl;
    myFile << "\\end{table}" << std::endl;

    myFile.close();
}

void printToPython(std::vector<long long> gpuRuntime, std::vector<long long> cpuRuntime){
    std::ofstream myFile("pythonProgram.py");

    myFile << "import matplotlib.pyplot as plt" << endl;
    myFile << "import matplotlib.ticker as ticker" << endl;
    myFile << "import numpy as np" << endl;
    myFile << endl;


}

//import matplotlib.pyplot as plt
//import matplotlib.ticker as ticker
//import numpy as np

//sizes = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000, 2048000, 4096000, 8192000, 16384000, 32768000, 65536000, 131072000, 262144000, 524288000]
//thrust = [86, 180, 93, 193, 384, 723, 1267, 1582, 2488, 3335, 3764, 4145, 4437, 3799, 3419, 3200, 2824, 3025, 3204, 3182]
//tbb = [180, 280, 185, 375, 558, 912, 1009, 1233, 1203, 1301, 1324, 1303, 1271, 1039, 793, 681, 651, 637, 662, 652]

//xpoints_1 = np.array([1000, 524288000/3])
//ypoints_1 = np.array([0, 3182])

//xpoints_2 = np.array([1000, 524288000/3])
//ypoints_2 = np.array([0, 652])

//fig1, ax1 = plt.subplots(figsize=(7, 4.5))
//ax1.plot(sizes, thrust, marker='s', linewidth=2, markersize=6, label='Thrust (GPU)')
//ax1.plot(sizes, tbb, marker='s', linewidth=2, markersize=6, label='TBb (CPU)')
//plt.plot(xpoints_1, ypoints_1)
//plt.plot(xpoints_2, ypoints_2)

//plt.axvline(x = 524288000/3, color = 'b', label = '(n/p)')

//ax1.set_xscale('log')
//ax1.set_xlabel('Problem Size (Number of Elements)', fontsize=11)
//ax1.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)
//ax1.set_title('GPU Thrust Sorting', fontsize=13, weight='bold')
//ax1.legend(frameon=True)
//ax1.grid(True, linestyle='--', alpha=0.6)
//ax1.tick_params(axis='both', labelsize=10)
//plt.tight_layout()
//plt.savefig('../../graphs/dataPartition1_4.pdf', bbox_inches='tight')