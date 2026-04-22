#include <vector>
#include <cstdlib>
#include <climits>
#include <cstdio>

extern double measuring_cuda_speed(int n, std::vector<long long> &data, int device);
extern double measuring_tbb_speed(int n, std::vector<long long> &data);

int main(){
    int totalSize=1000000000;
    int n=4000;
    int steps=18;
    int iterations=10;
    std::vector<double> speedCPU(steps);
    std::vector<double> speedGPU(steps);
    std::vector<double> speedGPU2(steps);
    std::vector<long long> data(totalSize);

    printf("creating randomized vector of size %d\n", totalSize);
    for(int a=0;a<totalSize;a++){
        data[a]=rand()%1000000000;
    }

    for(int a=0;a<steps;a++){
        for(int b=0;b<iterations;b++){
            speedCPU[a]=speedCPU[a]+measuring_tbb_speed(n, data);
        }
        n=n*2;
    }

    n=1000;
    for(int a=0;a<steps;a++){
        for(int b=0;b<iterations;b++){
            speedGPU[a]=speedGPU[a]+measuring_cuda_speed(n, data, 0);
        }
        n=n*2;
    }

    n=1000;
    for(int a=0;a<steps;a++){
        for(int b=0;b<iterations;b++){
            speedGPU2[a]=speedGPU2[a]+measuring_cuda_speed(n, data, 1);
        }
        n=n*2;
    }

    n=1000;
    printf("sizes = [");
    for(int a=1; a<=steps; a++){
        printf("%d%s", n, a<steps ? ", " : "");
        n=n*2;
    }
    printf("]\n");

    printf("GPU1 = [");
    for(int a=0; a<steps; a++){
        printf("%f%s", speedGPU[a]/iterations, a<steps-1 ? ", " : "");
    }
    printf("]\n");

    printf("GPU2 = [");
    for(int a=0;a<steps;a++){
        printf("%f%s", speedGPU2[a]/iterations, a<steps-1 ? ", " : "");
    }
    printf("]\n");

    printf("CPU = [");
    for(int a=0; a<steps; a++){
        printf("%f%s", speedCPU[a]/iterations, a<steps-1 ? ", " : "");
    }
    printf("]\n");

    printf("ratio1 = [");
    for(int a=0;a<steps;a++){
        printf("%f%s", (speedCPU[a] / speedGPU[a]), a<steps-1 ? ", " : "");
    }
    printf("]\n");

    printf("ratio2 = [");
    for(int a=0;a<steps;a++){
        printf("%f%s", (speedGPU[a]/speedGPU2[a]), a<steps-1 ? ", " : "");
    }
    printf("]\n");

    system("mkdir -p ../../python");

    FILE *fptr;
    fptr = fopen("../../python/dataPartition1.py", "w");
    fprintf(fptr, "import matplotlib.pyplot as plt\nimport matplotlib.ticker as ticker\n\n");
    n=1000;
    fprintf(fptr, "sizes = [");
    for(int a=1; a<=steps; a++){
        fprintf(fptr, "%d%s", n, a<steps ? ", " : "");
        n=n*2;
    }
    fprintf(fptr, "]\n");

    fprintf(fptr, "GPU1 = [");
    for(int a=0; a<steps; a++){
        fprintf(fptr, "%.0f%s", speedGPU[a]/iterations, a<steps-1 ? ", " : "");
    }
    fprintf(fptr, "]\n");

    fprintf(fptr, "GPU2 = [");
    for(int a=0;a<steps;a++){
        fprintf(fptr, "%.0f%s", speedGPU2[a]/iterations, a<steps-1 ? ", " : "");
    }
    fprintf(fptr, "]\n");

    fprintf(fptr, "CPU = [");
    for(int a=0; a<steps; a++){
        fprintf(fptr, "%.0f%s", speedCPU[a]/iterations, a<steps-1 ? ", " : "");
    }
    fprintf(fptr, "]\n");

    fprintf(fptr, "ratio1 = [");
    for(int a=0;a<steps;a++){
        fprintf(fptr, "%f%s", (speedCPU[a] / speedGPU[a]), a<steps-1 ? ", " : "");
    }
    fprintf(fptr, "]\n");

    fprintf(fptr, "ratio2 = [");
    for(int a=0;a<steps;a++){
        fprintf(fptr, "%f%s", (speedGPU[a]/speedGPU2[a]), a<steps-1 ? ", " : "");
    }
    fprintf(fptr, "]\n");

    fprintf(fptr,
            "fig1, ax1 = plt.subplots(figsize=(7, 4.5))\n"
            "ax1.plot(sizes, GPU1, marker='s', linewidth=2, markersize=6, label='Thrust (GPU)')\n"
            "ax1.set_xscale('log')\n"
            "ax1.set_xlabel('Problem Size (Number of Elements)', fontsize=11)\n"
            "ax1.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)\n"
            "ax1.set_title('CPU TBB Sorting', fontsize=13, weight='bold')\n"
            "ax1.legend(frameon=True)\n"
            "ax1.grid(True, linestyle='--', alpha=0.6)\n"
            "ax1.tick_params(axis='both', labelsize=10)\n"
            "plt.tight_layout()\n"
            "plt.savefig('../graphs/dataPartitionGPU1.pdf', bbox_inches='tight')\n"
            "\n\n"

            "fig2, ax2 = plt.subplots(figsize=(7, 4.5))\n"
            "ax2.plot(sizes, GPU2, marker='s', linewidth=2, markersize=6, label='Thrust (GPU)')\n"
            "ax2.set_xscale('log')\n"
            "ax2.set_xlabel('Problem Size (Number of Elements)', fontsize=11)\n"
            "ax2.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)\n"
            "ax2.set_title('CPU TBB Sorting', fontsize=13, weight='bold')\n"
            "ax2.legend(frameon=True)\n"
            "ax2.grid(True, linestyle='--', alpha=0.6)\n"
            "ax2.tick_params(axis='both', labelsize=10)\n"
            "plt.tight_layout()\n"
            "plt.savefig('../graphs/dataPartitionGPU1.pdf', bbox_inches='tight')\n"
            "\n\n"

            "fig3, ax3 = plt.subplots(figsize=(7, 4.5))\n"
            "ax3.plot(sizes, CPU, marker='o', linewidth=2, markersize=6, label='TBB (CPU)')\n"
            "ax3.set_xscale('log')\n"
            "ax3.set_xlabel('Problem Size (Number of Elements)', fontsize=11)\n"
            "ax3.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)\n"
            "ax3.set_title('GPU Thrust Sorting', fontsize=13, weight='bold')\n"
            "ax3.legend(frameon=True)\n"
            "ax3.grid(True, linestyle='--', alpha=0.6)\n"
            "ax3.tick_params(axis='both', labelsize=10)\n"
            "plt.tight_layout()\n"
            "plt.savefig('../graphs/dataPartitionCPU.pdf', bbox_inches='tight')\n"
            "\n\n"
            
            "fig4, ax4 = plt.subplots(figsize=(7, 4.5))\n"
            "ax4.plot(sizes, ratio1, marker='o', linewidth=2, markersize=6, label='ratio between CPU sort & GPU sort')\n"
            "ax4.set_xscale('log')\n"
            "ax4.set_xlabel('Problem Size (Number of Elements)', fontsize=11)\n"
            "ax4.set_ylabel('Ratio of Processing Speed', fontsize=11)\n"
            "ax4.set_title('Ratio of CPU to GPU', fontsize=13, weight='bold')\n"
            "ax4.legend(frameon=True)\n"
            "ax4.grid(True, linestyle='--', alpha=0.6)\n"
            "ax4.tick_params(axis='both', labelsize=10)\n"
            "plt.tight_layout()\n"
            "plt.savefig('../graphs/dataPartitionRatioCPUandGPU.pdf', bbox_inches='tight')\n"
            "\n\n"

            "fig5, ax5 = plt.subplots(figsize=(7, 4.5))\n"
            "ax5.plot(sizes, ratio2, marker='o', linewidth=2, markersize=6, label='ratio between GPU1 and GPU2')\n"
            "ax5.set_xscale('log')\n"
            "ax5.set_xlabel('Problem Size (Number of Elements)', fontsize=11)\n"
            "ax5.set_ylabel('Ratio of Processing Speed', fontsize=11)\n"
            "ax5.set_title('Ratio of GPU1 to GPU2', fontsize=13, weight='bold')\n"
            "ax5.legend(frameon=True)\n"
            "ax5.grid(True, linestyle='--', alpha=0.6)\n"
            "ax5.tick_params(axis='both', labelsize=10)\n"
            "plt.tight_layout()\n"
            "plt.savefig('../graphs/dataPartitionRatioGPU1andGPU2.pdf', bbox_inchest='tight')\n"
        );
}
