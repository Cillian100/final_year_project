import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sizes = [256, 65536, 16777216, 4294967296, 1099511627776, 281474976710656, 72057594037927936, 18446744073709551616]
ratio_10000000 = [0.152, 0.126087, 0.165746, 0.181818, 0.26875, 0.354839, 0.46875, 0.621622]
ratio_100000000 = [0.479393, 0.562791, 0.606715, 0.72, 0.918301, 1.06227, 1.37788, 1.71098]
ratio_1000000000 = [0.806864, 0.843123, 0.891151, 0.988545, 1.07421, 1.38055, 1.52671, 2.11994]

fig1, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(sizes, ratio_100000000, marker='s', linewidth=2, markersize=6, label='Input of Length 100000000')
ax1.plot(sizes, ratio_1000000000, marker='s', linewidth=2, markersize=6, label='Input of Length 1000000000')
ax1.set_xscale('log')
ax1.set_xlabel('Range of Input Elements', fontsize=11)
ax1.set_ylabel('Algorithm Runtime in MS', fontsize=11)
ax1.set_title('MSD vs LSD Radix Sort ratio', fontsize=11)
ax1.legend(frameon=True)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig('../../graphs/iterations2_3.pdf', bbox_inches='tight')

