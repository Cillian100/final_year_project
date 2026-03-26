#include <vector>
#include <cstdlib>
#include <climits>
#include <cstdio>

extern double measuring_cuda_speed(int n, std::vector<long long> &data);
extern double measuring_tbb_speed(int n, std::vector<long long> &data);

int main(){
    //int totalSize=262144000;
    int totalSize=1000000000;
    int n=1000;
    int steps=20;
    std::vector<double> speedCPU(steps);
    std::vector<double> speedGPU(steps); 
    std::vector<long long> data(totalSize);

    printf("creating randomized vector of size %d\n", totalSize);
    for(int a=0;a<totalSize;a++){
        data[a]=rand()%LLONG_MAX;
    }

    for(int a=0;a<steps;a++){
        speedCPU[a]=measuring_tbb_speed(n, data);
        n=n*2;
    }

    n=1000;
    for(int a=0;a<steps;a++){
        speedGPU[a]=measuring_cuda_speed(n, data);
        n=n*2;
    }

    n=1000;
    printf("sizes = [");
    for(int a=1; a<=steps; a++){
        printf("%d%s", n, a<steps ? ", " : "");
        n=n*2;
    }
    printf("]\n");

    printf("thrust = [");
    for(int a=0; a<steps; a++){
        printf("%.0f%s", speedGPU[a], a<steps-1 ? ", " : "");
    }
    printf("]\n");

    printf("tbb = [");
    for(int a=0; a<steps; a++){
        printf("%.0f%s", speedCPU[a], a<steps-1 ? ", " : "");
    }
    printf("]\n");

    FILE *fptr;
    fptr = fopen("test.py", "w");
    fprintf(fptr, "import matplotlib.pyplot as plt\nimport matplotlib.ticker as ticker\n\n");
    n=1000;
    fprintf(fptr, "sizes = [");
    for(int a=1; a<=steps; a++){
        fprintf(fptr, "%d%s", n, a<steps ? ", " : "");
        n=n*2;
    }
    fprintf(fptr, "]\n");

    fprintf(fptr, "thrust = [");
    for(int a=0; a<steps; a++){
        fprintf(fptr, "%.0f%s", speedGPU[a], a<steps-1 ? ", " : "");
    }
    fprintf(fptr, "]\n");

    fprintf(fptr, "tbb = [");
    for(int a=0; a<steps; a++){
        fprintf(fptr, "%.0f%s", speedCPU[a], a<steps-1 ? ", " : "");
    }
    fprintf(fptr, "]\n");

    fprintf(fptr,
                "fig, ax = plt.subplots(figsize=(7, 4.5))\n"
                "\n"
                "# Plot data with clearer styling\n"
                "ax.plot(sizes, thrust, marker='s', linewidth=2, markersize=6, label='Thrust (GPU)')\n"
                "ax.plot(sizes, tbb, marker='o', linewidth=2, markersize=6, label='TBB (CPU)')\n"
                "\n"
                "# Labels and title\n"
                "ax.set_xlabel('Problem Size (Number of Elements)', fontsize=11)\n"
                "ax.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)\n"
                "ax.set_title('Sorting Performance Comparison: GPU vs CPU', fontsize=13, weight='bold')\n"
                "\n"
                "# Improve legend and grid\n"
                "ax.legend(frameon=True)\n"
                "ax.grid(True, linestyle='--', alpha=0.6)\n"
                "\n"
                "# Optional: format ticks for readability\n"
                "ax.tick_params(axis='both', labelsize=10)\n"
                "\n"
                "# Tight layout and save\n"
                "plt.tight_layout()\n"
                "plt.savefig('sorting_performance.pdf', bbox_inches='tight')\n"
            );
}
