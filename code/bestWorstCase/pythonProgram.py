import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sizes = [100000000, 200000000, 300000000, 400000000, 500000000]
worstGPU = [258, 515, 774, 1028, 1289]
bestGPU = [270, 503, 773, 1028, 1291]
worstCPU = [460, 816, 1145, 1487, 1833]
bestCPU = [460, 816, 1145, 1487, 1833]
fig1, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(sizes, worstGPU, marker='s', linewidth=2, markersize=6, label='worse case GPU')
ax1.plot(sizes, bestGPU, marker='s', linewidth=2, markersize=6, label='best case GPU')
ax1.plot(sizes, worstCPU, marker='s', linewidth=2, markersize=6, label='worse case CPU')
ax1.plot(sizes, bestCPU, marker='s', linewidth=2, markersize=6, label='best case CPU')
ax1.set_xlabel('Problem Size (Number of Elements)', fontsize=11)
ax1.set_ylabel('Runtime (ms)', fontsize=11)
ax1.set_title('Comparision of sorted vs unsorted array in radix sort', fontsize=13, weight='bold')
ax1.legend(frameon=True)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.tick_params(axis='both', labelsize=10)
plt.tight_layout() 
plt.savefig('../../graphs/bestWorstCase.pdf', bbox_inches='tight') 
