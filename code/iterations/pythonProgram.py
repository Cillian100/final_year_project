import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sizes = [256, 65536, 16777216, 4294967296, 1099511627776, 281474976710656, 72057594037927936, 18446744073709551616]
gpuRuntime = [234, 246, 254, 273, 282, 293, 288, 312]
cpuRuntime = [447, 425, 414, 377, 321, 273, 222, 169]
fig1, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(sizes, cpuRuntime, marker='s', linewidth=2, markersize=6, label='CPU MSD Radix Sort')
ax1.plot(sizes, gpuRuntime, marker='s', linewidth=2, markersize=6, label='GPU LSD Radix Sort') 
ax1.set_xscale('log')
ax1.set_xlabel('Range of Input Elements', fontsize=11)
ax1.set_ylabel('Algorithm Runtime in MS', fontsize=11)
ax1.set_title('MSD vs LSD Radix Sort', fontsize=11)
ax1.legend(frameon=True)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig('../../graphs/iterations1.pdf', bbox_inches='tight')

