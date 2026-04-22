import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sizes = [256, 65536, 16777216, 4294967296, 1099511627776, 281474976710656, 72057594037927936, 18446744073709551616]
thrust = [221, 231, 242, 260, 269, 278, 287, 296]
tbb = [559, 635, 690, 710, 735, 753, 751, 768]

fig1, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(sizes, thrust, marker='s', linewidth=2, markersize=6, label='Thrust (GPU)')
ax1.plot(sizes, tbb, marker='s', linewidth=2, markersize=6, label='TBb (CPU)')

ax1.set_xscale('log')
ax1.set_xlabel('Problem Size (Number of Elements)', fontsize=11)
ax1.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)
ax1.set_title('GPU Thrust Sorting', fontsize=13, weight='bold')
ax1.legend(frameon=True)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig('../../graphs/iterations.pdf', bbox_inches='tight')
