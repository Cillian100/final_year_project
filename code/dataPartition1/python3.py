import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


sizes = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000, 2048000, 4096000, 8192000, 16384000, 32768000, 65536000, 131072000, 262144000, 524288000]
thrust = [86, 180, 93, 193, 384, 723, 1267, 1582, 2488, 3335, 3764, 4145, 4437, 3799, 3419, 3200, 2824, 3025, 3204, 3182]
tbb = [180, 280, 185, 375, 558, 912, 1009, 1233, 1203, 1301, 1324, 1303, 1271, 1039, 793, 681, 651, 637, 662, 652]

xpoints_1 = np.array([1000, 524288000/3])
ypoints_1 = np.array([0, 3182])

xpoints_2 = np.array([1000, 524288000/3])
ypoints_2 = np.array([0, 652])

fig1, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(sizes, thrust, marker='s', linewidth=2, markersize=6, label='Thrust (GPU)')
ax1.plot(sizes, tbb, marker='s', linewidth=2, markersize=6, label='TBb (CPU)')

plt.plot(xpoints_1, ypoints_1)
plt.plot(xpoints_2, ypoints_2)

optimal_x_gpu = 234133230
optimal_speed_gpu = thrust[-1]

optimal_slope = optimal_speed_gpu / optimal_x_gpu

xpoints_M = np.array([1000, 524288000/3])
ypoints_M = np.array([optimal_slope * 1000, optimal_slope * (524288000/3)])

plt.plot(xpoints_M, ypoints_M, color='green', linestyle='--', label='Optimal line M')

plt.axvline(x = 524288000/3, color = 'b', label = '(n/p)')

ax1.set_xscale('log')
ax1.set_xlabel('Problem Size (Number of Elements)', fontsize=11)
ax1.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)
ax1.set_title('GPU Thrust Sorting', fontsize=13, weight='bold')
ax1.legend(frameon=True)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig('dataPartition1_5.pdf', bbox_inches='tight')