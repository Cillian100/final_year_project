#include <vector>
#include <cstdlib>
#include <climits>
#include <cstdio>

extern double measuring_cuda_speed(int n, std::vector<long long> &data);
extern double measuring_tbb_speed(int n, std::vector<long long> &data);

int main()
{
    int totalSize = 20000000;
    long long n = 100;
    int steps = 16;
    std::vector<double> speedCPU(steps);
    std::vector<double> speedGPU(steps);
    std::vector<double> speedRatio(steps);

    std::vector<long long> data(totalSize);
    printf("creating randomized vector of size %d\n", totalSize);

    for (int a = 0; a < steps; a++)
    {
        for (int b = 0; b < totalSize; b++)
        {
            data[b] = rand() % n;
        }
        printf("poop %lld\n", n);
        for(int b=0;b<5;b++){
            speedCPU[a] = speedCPU[a] + measuring_tbb_speed(totalSize, data);
        }

        for(int b=0;b<5;b++){
            speedGPU[a] = speedGPU[a] + measuring_cuda_speed(totalSize, data);
        }
        n = n * 10;
    }

    n = 100;
    for (int a = 0; a < steps; a++)
    {
        speedRatio[a] = speedCPU[a] / speedGPU[a];
        printf("%d %f %f %f\n", a, speedCPU[a], speedGPU[a], speedRatio[a]);
    }

    FILE *fptr;
    fptr = fopen("test.py", "w");
    fprintf(fptr, "import matplotlib.pyplot as plt\nimport matplotlib.ticker as ticker\n\n");
    n = 100;
    fprintf(fptr, "sizes = [");
    for (int a = 0; a < steps; a++)
    {
        fprintf(fptr, "%lld%s", n, a < steps - 1 ? ", " : "");
        n = n * 10;
    }
    fprintf(fptr, "]\n");

    fprintf(fptr, "thrust = [");
    for (int a = 0; a < steps; a++)
    {
        fprintf(fptr, "%.0f%s", (speedGPU[a]/5), a < steps - 1 ? ", " : "");
    }
    fprintf(fptr, "]\n");

    fprintf(fptr, "tbb = [");
    for (int a = 0; a < steps; a++)
    {
        fprintf(fptr, "%.0f%s", (speedCPU[a]/5), a < steps - 1 ? ", " : "");
    }
    fprintf(fptr, "]\n");

    fprintf(fptr, "ratio = [");
    for(int a=0;a<steps;a++){
        fprintf(fptr, "%.3f%s", (speedRatio[a]), a < steps - 1 ? ", " : "");
    }
    fprintf(fptr, "]\n");

    fprintf(fptr,
            "fig1, ax1 = plt.subplots(figsize=(7, 4.5))\n"
            "ax1.plot(sizes, thrust, marker='s', linewidth=2, markersize=6, label='Thrust (GPU)')\n"
            "ax1.set_xscale('log')\n"
            "ax1.set_xlabel('Problem Size (Number of Elements)', fontsize=11)\n"
            "ax1.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)\n"
            "ax1.set_title('CPU TBB Sorting', fontsize=13, weight='bold')\n"
            "ax1.legend(frameon=True)\n"
            "ax1.grid(True, linestyle='--', alpha=0.6)\n"
            "ax1.tick_params(axis='both', labelsize=10)\n"
            "plt.tight_layout()\n"
            "plt.savefig('dataPartition2_1.pdf', bbox_inches='tight')\n"
            "\n\n"
            "fig2, ax2 = plt.subplots(figsize=(7, 4.5))\n"
            "ax2.plot(sizes, tbb, marker='o', linewidth=2, markersize=6, label='TBB (CPU)')\n"
            "ax2.set_xscale('log')\n"
            "ax2.set_xlabel('Problem Size (Number of Elements)', fontsize=11)\n"
            "ax2.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)\n"
            "ax2.set_title('GPU Thrust Sorting', fontsize=13, weight='bold')\n"
            "ax2.legend(frameon=True)\n"
            "ax2.grid(True, linestyle='--', alpha=0.6)\n"
            "ax2.tick_params(axis='both', labelsize=10)\n"
            "plt.tight_layout()\n"
            "plt.savefig('dataPartition2_2.pdf', bbox_inches='tight')\n"
            "\n\n"
            "fig3, ax3 = plt.subplots(figsize=(7, 4.5))\n"
            "ax3.plot(sizes, ratio, marker='o', linewidth=2, markersize=6, label='ratio between CPU sort & GPU sort')\n"
            "ax3.set_xscale('log')\n"
            "ax3.set_xlabel('Problem Size (Number of Elements)', fontsize=11)\n"
            "ax3.set_ylabel('Ratio of Processing Speed', fontsize=11)\n"
            "ax3.set_title('Ratio of CPU to GPU', fontsize=13, weight='bold')\n"
            "ax3.legend(frameon=True)\n"
            "ax3.grid(True, linestyle='--', alpha=0.6)\n"
            "ax3.tick_params(axis='both', labelsize=10)\n"
            "plt.tight_layout()\n"
            "plt.savefig('dataPartition2_3.pdf', bbox_inches='tight')\n"
            "\n\n"
        );
}
