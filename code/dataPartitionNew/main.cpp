#include <math.h>
#include <vector>
#include <stdio.h>
#include <random>
#include <algorithm>
#include <pthread.h>
#include <fstream>
#include <iostream>
#include <filesystem>

 
extern double measuring_thrust_speed(std::vector<long long> &data, int device, long long size);
extern double measuring_raduls_speed(std::vector<long long> &data, long long size);
extern double measuring_tbb_speed(std::vector<long long> &data, long long n);
void generatePythonFile(std::vector<double> &speedGPU, std::vector<double> &speedGPU2,
                        std::vector<double> &speedCPUraduls, std::vector<double> &speedCPU2intel,
                        int steps, int iterations, int n2);
 
void function(std::vector<long long> data, long long range){
    long long totalSize = 1000000000;
    long long n2        = 10000;
    int steps     = 17;
    int iterations = 10;
 
    std::vector<double>    speedCPUraduls(steps, 0.0);
    std::vector<double>    speedCPU2intel(steps, 0.0);
    std::vector<double>    speedGPU(steps,       0.0);
    std::vector<double>    speedGPU2(steps,      0.0);

    // CPU RADULS
    int n = n2;
    for(int a = 0; a < steps; a++){
        for(int b = 0; b < iterations; b++)
            speedCPUraduls[a] += measuring_raduls_speed(data, n);
        n *= 2;
    }
 
    // CPU Intel TBB
    n = n2;
    for(int a = 0; a < steps; a++){
        for(int b = 0; b < iterations; b++)
            speedCPU2intel[a] += measuring_tbb_speed(data, n);
        n *= 2;
    }
 
    // GPU1
    n = n2;
    for(int a = 0; a < steps; a++){
        for(int b = 0; b < iterations; b++)
            speedGPU[a] += measuring_thrust_speed(data, 0, n);
        n *= 2;
    }
 
    // GPU2
    n = n2;
    for(int a = 0; a < steps; a++){
        for(int b = 0; b < iterations; b++)
            speedGPU2[a] += measuring_thrust_speed(data, 1, n);
        n *= 2;
    }
 
    // Print results to stdout
    n = n2;
    printf("sizes = [");
    for(int a = 1; a <= steps; a++){
        printf("%d%s", n, a < steps ? ", " : "");
        n *= 2;
    }
    printf("]\n");
 
    printf("GPU1 = [");
    for(int a = 0; a < steps; a++)
        printf("%f%s", speedGPU[a]/iterations, a < steps-1 ? ", " : "");
    printf("]\n");
 
    printf("GPU2 = [");
    for(int a = 0; a < steps; a++)
        printf("%f%s", speedGPU2[a]/iterations, a < steps-1 ? ", " : "");
    printf("]\n");
 
    printf("CPU1 = [");
    for(int a = 0; a < steps; a++)
        printf("%f%s", speedCPUraduls[a]/iterations, a < steps-1 ? ", " : "");
    printf("]\n");
 
    printf("CPU2 = [");
    for(int a = 0; a < steps; a++)
        printf("%f%s", speedCPU2intel[a]/iterations, a < steps-1 ? ", " : "");
    printf("]\n");
 
    printf("ratio1 = [");
    for(int a = 0; a < steps; a++)
        printf("%f%s", speedCPUraduls[a]/speedGPU[a], a < steps-1 ? ", " : "");
    printf("]\n");
 
    printf("ratio2 = [");
    for(int a = 0; a < steps; a++)
        printf("%f%s", speedGPU[a]/speedGPU2[a], a < steps-1 ? ", " : "");
    printf("]\n");
 
    //generatePythonFile(speedGPU, speedGPU2, speedCPUraduls, speedCPU2intel, steps, iterations, n2);
 
    return;
}

int main(){
    long long totalSize = 1000000000;
    std::vector<long long> data(totalSize);  // allocate once

    std::vector<long long> ranges = {256, 65536, 16777216, 4294967296,
                                     1099511627776, 281474976710656,
                                     72057594037927936, INT64_MAX};

    std::mt19937_64 rng(std::random_device{}());

    for(int a = 4; a < 8; a++){
        // Regenerate values for this range in-place — no reallocation
        std::uniform_int_distribution<long long> dist(0, ranges[a]);
        std::generate(data.begin(), data.end(), [&](){ return dist(rng); });

        printf("RANGE %lld\n", ranges[a]);
        function(data, ranges[a]);
        printf("\n\n");
    }
}
void generatePythonFile(std::vector<double> &speedGPU, std::vector<double> &speedGPU2,
                        std::vector<double> &speedCPUraduls, std::vector<double> &speedCPU2intel,
                        int steps, int iterations, int n2){
    std::filesystem::create_directories("../../python");
    std::ofstream f("../../python/dataPartitionNew1.py");
 
    f << "import matplotlib.pyplot as plt\n"
      << "import matplotlib.ticker as ticker\n\n";
 
    // sizes
    int n = n2;
    f << "sizes = [";
    for(int a = 1; a <= steps; a++){
        f << n << (a < steps ? ", " : "");
        n *= 2;
    }
    f << "]\n";
 
    // data arrays
    f << "GPU1 = [";
    for(int a = 0; a < steps; a++)
        f << std::fixed << speedGPU[a]/iterations << (a < steps-1 ? ", " : "");
    f << "]\n";
 
    f << "GPU2 = [";
    for(int a = 0; a < steps; a++)
        f << std::fixed << speedGPU2[a]/iterations << (a < steps-1 ? ", " : "");
    f << "]\n";
 
    f << "CPU1 = [";
    for(int a = 0; a < steps; a++)
        f << std::fixed << speedCPUraduls[a]/iterations << (a < steps-1 ? ", " : "");
    f << "]\n";
 
    f << "CPU2 = [";
    for(int a = 0; a < steps; a++)
        f << std::fixed << speedCPU2intel[a]/iterations << (a < steps-1 ? ", " : "");
    f << "]\n";
 
    f << "ratio1 = [";
    for(int a = 0; a < steps; a++)
        f << std::fixed << speedCPUraduls[a]/speedGPU[a] << (a < steps-1 ? ", " : "");
    f << "]\n";
 
    f << "ratio2 = [";
    for(int a = 0; a < steps; a++)
        f << std::fixed << speedGPU[a]/speedGPU2[a] << (a < steps-1 ? ", " : "");
    f << "]\n\n";
 
    // Fig1 - GPU1
    f << "fig1, ax1 = plt.subplots(figsize=(7, 4.5))\n"
      << "ax1.plot(sizes, GPU1, marker='s', linewidth=2, markersize=6, label='Thrust GPU1')\n"
      << "ax1.set_xscale('log')\n"
      << "ax1.set_xlabel('Problem Size (Number of Elements)', fontsize=11)\n"
      << "ax1.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)\n"
      << "ax1.set_title('GPU1 Thrust Sorting', fontsize=13, weight='bold')\n"
      << "ax1.legend(frameon=True)\n"
      << "ax1.grid(True, linestyle='--', alpha=0.6)\n"
      << "ax1.tick_params(axis='both', labelsize=10)\n"
      << "plt.tight_layout()\n"
      << "plt.savefig('../graphs/dataPartitionNew/dataPartitionGPU1.pdf', bbox_inches='tight')\n\n";
 
    // Fig2 - GPU2
    f << "fig2, ax2 = plt.subplots(figsize=(7, 4.5))\n"
      << "ax2.plot(sizes, GPU2, marker='s', linewidth=2, markersize=6, label='Thrust GPU2')\n"
      << "ax2.set_xscale('log')\n"
      << "ax2.set_xlabel('Problem Size (Number of Elements)', fontsize=11)\n"
      << "ax2.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)\n"
      << "ax2.set_title('GPU2 Thrust Sorting', fontsize=13, weight='bold')\n"
      << "ax2.legend(frameon=True)\n"
      << "ax2.grid(True, linestyle='--', alpha=0.6)\n"
      << "ax2.tick_params(axis='both', labelsize=10)\n"
      << "plt.tight_layout()\n"
      << "plt.savefig('../graphs/dataPartitionNew/dataPartitionGPU2.pdf', bbox_inches='tight')\n\n";
 
    // Fig3 - CPU1 (RADULS)
    f << "fig3, ax3 = plt.subplots(figsize=(7, 4.5))\n"
      << "ax3.plot(sizes, CPU1, marker='o', linewidth=2, markersize=6, label='RADULS (CPU)')\n"
      << "ax3.set_xscale('log')\n"
      << "ax3.set_xlabel('Problem Size (Number of Elements)', fontsize=11)\n"
      << "ax3.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)\n"
      << "ax3.set_title('CPU RADULS Sorting', fontsize=13, weight='bold')\n"
      << "ax3.legend(frameon=True)\n"
      << "ax3.grid(True, linestyle='--', alpha=0.6)\n"
      << "ax3.tick_params(axis='both', labelsize=10)\n"
      << "plt.tight_layout()\n"
      << "plt.savefig('../graphs/dataPartitionNew/dataPartitionCPU1.pdf', bbox_inches='tight')\n\n";
 
    // Fig4 - CPU2 (Intel TBB)
    f << "fig4, ax4 = plt.subplots(figsize=(7, 4.5))\n"
      << "ax4.plot(sizes, CPU2, marker='o', linewidth=2, markersize=6, label='Intel TBB (CPU)')\n"
      << "ax4.set_xscale('log')\n"
      << "ax4.set_xlabel('Problem Size (Number of Elements)', fontsize=11)\n"
      << "ax4.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)\n"
      << "ax4.set_title('CPU Intel TBB Sorting', fontsize=13, weight='bold')\n"
      << "ax4.legend(frameon=True)\n"
      << "ax4.grid(True, linestyle='--', alpha=0.6)\n"
      << "ax4.tick_params(axis='both', labelsize=10)\n"
      << "plt.tight_layout()\n"
      << "plt.savefig('../graphs/dataPartitionNew/dataPartitionCPU2.pdf', bbox_inches='tight')\n\n";
 
    // Fig5 - ratio CPU vs GPU1
    f << "fig5, ax5 = plt.subplots(figsize=(7, 4.5))\n"
      << "ax5.plot(sizes, ratio1, marker='o', linewidth=2, markersize=6, label='ratio CPU1 / GPU1')\n"
      << "ax5.set_xscale('log')\n"
      << "ax5.set_xlabel('Problem Size (Number of Elements)', fontsize=11)\n"
      << "ax5.set_ylabel('Ratio of Processing Speed', fontsize=11)\n"
      << "ax5.set_title('Ratio of CPU to GPU', fontsize=13, weight='bold')\n"
      << "ax5.legend(frameon=True)\n"
      << "ax5.grid(True, linestyle='--', alpha=0.6)\n"
      << "ax5.tick_params(axis='both', labelsize=10)\n"
      << "plt.tight_layout()\n"
      << "plt.savefig('../graphs/dataPartitionNew/dataPartitionRatioCPUandGPU.pdf', bbox_inches='tight')\n\n";
 
    // Fig6 - ratio GPU1 vs GPU2
    f << "fig6, ax6 = plt.subplots(figsize=(7, 4.5))\n"
      << "ax6.plot(sizes, ratio2, marker='o', linewidth=2, markersize=6, label='ratio GPU1 / GPU2')\n"
      << "ax6.set_xscale('log')\n"
      << "ax6.set_xlabel('Problem Size (Number of Elements)', fontsize=11)\n"
      << "ax6.set_ylabel('Ratio of Processing Speed', fontsize=11)\n"
      << "ax6.set_title('Ratio of GPU1 to GPU2', fontsize=13, weight='bold')\n"
      << "ax6.legend(frameon=True)\n"
      << "ax6.grid(True, linestyle='--', alpha=0.6)\n"
      << "ax6.tick_params(axis='both', labelsize=10)\n"
      << "plt.tight_layout()\n"
      << "plt.savefig('../graphs/dataPartitionNew/dataPartitionRatioGPU1andGPU2.pdf', bbox_inches='tight')\n";
 
    f.close();
    printf("Python file written to ../../python/dataPartition1.py\n");
}