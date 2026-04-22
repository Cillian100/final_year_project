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
void printToLatex(const std::vector<std::vector<long long>> &myVec1,
                  const std::vector<std::vector<long long>> &myVec2,
                  const std::string &information,
                  std::ofstream &myFile);

typedef struct {
    int device;
    const std::vector<long long> *data;
    int startingPoint;
    int endingPoint;
    long int runtime;
} GPU_struct;

typedef struct {
    const std::vector<long long> *data;
    int startingPoint;
    int endingPoint;
    long int runtime;
} CPU_struct;

void* GPU_sort(void* arg) {
    GPU_struct *s = (GPU_struct *)arg;
    s->runtime = 0;

    // Allocate scratch buffer once and reuse across all 20 iterations
    std::vector<long long> mydata;
    mydata.reserve(s->data->size());

    for (int a = 0; a < 5; a++) {
        mydata = *s->data;
        s->runtime += thrust_sorting(s->startingPoint, s->endingPoint, s->device, mydata);
    }
    return NULL;
}

void* CPU_sort(void* arg) {
    CPU_struct *s = (CPU_struct *)arg;
    s->runtime = 0;

    // Allocate scratch buffer once and reuse across all 20 iterations
    std::vector<long long> mydata;
    mydata.reserve(s->data->size());

    for (int a = 0; a < 5; a++) {
        mydata = *s->data;
        s->runtime += tbb_sorting(s->startingPoint, s->endingPoint, mydata);
    }
    return NULL;
}

std::vector<long long> benchmark(const std::vector<long long> &dataPartition, int length) {
    std::vector<long long> data;
    data.reserve(length);
    printf("generating dataset\n");
    for (long long a = 0; a < length; a++) {
        data.push_back(rand() % 1000000000);
    }

    int first  = 0;
    int second = dataPartition.at(0);
    int third  = dataPartition.at(1) + second;
    int fourth = dataPartition.at(2) + third;

    pthread_t thread1, thread2, thread3;
    GPU_struct gpu1 = {0, &data, first,  second};
    GPU_struct gpu2 = {1, &data, second, third};
    CPU_struct cpu  = {   &data, third,  fourth};

    printf("sorting\n");
    pthread_create(&thread1, NULL, GPU_sort, &gpu1);
    pthread_create(&thread2, NULL, GPU_sort, &gpu2);
    pthread_create(&thread3, NULL, CPU_sort, &cpu);
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    pthread_join(thread3, NULL);
    printf("finished sorting\n");

    long int runtimes[3]  = {gpu1.runtime, gpu2.runtime, cpu.runtime};
    long int maxRuntime   = *std::max_element(runtimes, runtimes + 3);
    long int minRuntime   = *std::min_element(runtimes, runtimes + 3);
    long int spread       = maxRuntime - minRuntime;

    printf("\nBenchmark:\n");
    printf("GPU 1:   %ld ms\n", gpu1.runtime / 5);
    printf("GPU 2:   %ld ms\n", gpu2.runtime / 5);
    printf("CPU:     %ld ms\n", cpu.runtime  / 5);
    printf("Spread:  %ld ms\n", spread       / 5);
    printf("Balance: %.1f%%\n", (1.0 - (double)spread / maxRuntime) * 100.0);

    return {
        (long long)length,
        gpu1.runtime / 5,
        gpu2.runtime / 5,
        cpu.runtime  / 5,
        spread       / 5
    };
}

void warmup() {
    std::vector<long long> dummy(1000);
    thrust_sorting(0, 1000, 0, dummy);
    thrust_sorting(0, 1000, 1, dummy);
    tbb_sorting(0, 1000, dummy);
}

std::vector<long long> benchmark_partition(long long length) {
    long long third = length / 3;
    std::vector<long long> result = {third, third, length - 2 * third};
    return result;
}

int main() {
    long long length = 100000000;
    int iteration=3;
    std::vector<std::vector<long long>> myVec1;
    std::vector<std::vector<long long>> myVec2;

    // Pre-allocate to avoid incremental reallocations
    myVec1.reserve(iterations);
    myVec2.reserve(iterations);

    std::ofstream myFile("filename.txt");

    for (int a = 0; a < iterations; a++) {
        std::vector<long long> dataPartition1 = functional_partition(length);
        warmup();
        myVec1.push_back(benchmark(dataPartition1, length));
        length += 100000000;
    }

    length = 100000000;
    for (int a = 0; a < iterations; a++) {
        std::vector<long long> dataPartition2 = benchmark_partition(length);
        warmup();
        myVec2.push_back(benchmark(dataPartition2, length));
        length += 100000000;
    }

    printToLatex(myVec1, myVec2, "poop", myFile);
    myFile.close();
}

void printToLatex(const std::vector<std::vector<long long>> &myVec1,
                  const std::vector<std::vector<long long>> &myVec2,
                  const std::string &information,
                  std::ofstream &myFile) {
    auto writeTable = [&](const std::vector<std::vector<long long>> &myVec, const std::string &caption) {
        myFile << "\\begin{table}[h]\n"
               << "\\centering\n"
               << "\\caption{" << caption << "}\n"
               << "\\vspace{4pt}\n"
               << "\\begin{tabular}{lcccccc}\n"
               << "\\toprule\n"
               << "\\textbf{Array Size} & \\textbf{GPU 1 (ms)} & \\textbf{GPU 2 (ms)} & "
                  "\\textbf{CPU (ms)} & \\textbf{Spread (ms)} & \\textbf{Balance} \\\\\n"
               << "\\midrule\n";

        for (const auto &row : myVec) {
            for (int b = 0; b < 4; b++) {
                myFile << row.at(b) << " & ";
            }
            myFile << row.at(4) << "\\% \\\\\n";
        }

        myFile << "\\bottomrule\n"
               << "\\end{tabular}\n"
               << "\\end{table}\n";
    };

    writeTable(myVec1, "Runtime analysis of functional model in (ms)");
    writeTable(myVec2, "Runtime analysis of benchmark model in (ms)");
}