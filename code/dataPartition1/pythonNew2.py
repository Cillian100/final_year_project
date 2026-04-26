import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sizes = [10000, 20000, 40000, 80000, 160000, 320000, 640000, 1280000, 2560000,
         5120000, 10240000, 20480000, 40960000, 81920000, 163840000, 327680000,
         655360000, 1310720000, 2621440000]

gpu = [11754.4, 104297, 202244, 314787, 401397, 597073, 767086, 858627, 911662,
       844524, 787853, 806201, 817984, 824820, 826178, 824577, 817779, 820291, 832356]

cpu = [237.606, 576.079, 1199.88, 1895.36, 4044.97, 8455.48, 13075.1, 27146, 46933.6,
       75044.4, 126984, 215165, 290582, 409874, 407637, 591910, 525760, 603322, 620562]

n = 1310720000
p = 3
n_over_p = n / p

gpu_at_np = np.interp(n_over_p, sizes, gpu)
cpu_at_np = np.interp(n_over_p, sizes, cpu)

xpoints_1 = np.array([10000, n_over_p])
ypoints_1 = np.array([0, gpu_at_np])

xpoints_2 = np.array([10000, n_over_p])
ypoints_2 = np.array([0, cpu_at_np])

gpu_allocation = 641064918  # paste GPU workload from C++ printf output

optimal_slope = np.interp(gpu_allocation, sizes, gpu) / gpu_allocation

xpoints_M = np.array([10000, n_over_p])
ypoints_M = np.array([optimal_slope * 10000, optimal_slope * n_over_p])

fig1, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(sizes, gpu, marker='s', linewidth=2, markersize=6, label='Thrust (GPU)')
ax1.plot(sizes, cpu, marker='s', linewidth=2, markersize=6, label='TBb (CPU)')
ax1.plot(xpoints_1, ypoints_1)
ax1.plot(xpoints_2, ypoints_2)
ax1.plot(xpoints_M, ypoints_M, color='green', linestyle='--', label='Optimal line M')
ax1.axvline(x=n_over_p, color='b', label='(n/p)')

ax1.set_xscale('log')
ax1.set_xlabel('Problem Size (Number of Elements)', fontsize=11)
ax1.set_ylabel('Elements Sorted per Millisecond', fontsize=11)
ax1.set_title('GPU Thrust Sorting', fontsize=13, weight='bold')
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax1.legend(frameon=True)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig('graph.pdf', bbox_inches='tight')