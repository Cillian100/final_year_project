import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sizes = [256, 65536, 16777216, 4294967296, 1099511627776, 281474976710656, 72057594037927936, 18446744073709551616]
ratio_1 = [0.127273, 0.145749, 0.177885, 0.201031, 0.253165, 0.315385, 0.424242, 0.558442]
ratio_2 = [0.215753, 0.244444, 0.284519, 0.336449, 0.402174, 0.513514, 0.684211, 0.860215]
ratio_3 = [0.383387, 0.418919, 0.475836, 0.610619, 0.708543, 0.868263, 1.073529, 1.525253]
ratio_4 = [0.470588, 0.532637, 0.582873, 0.692547, 0.845324, 1.017094, 1.306818, 1.801418]
ratio_5 = [0.572358, 0.636519, 0.686380, 0.804642, 0.972603, 1.172775, 1.554422, 2.091304]
ratio_6 = [0.572358, 0.636519, 0.686380, 0.804642, 0.972603, 1.172775, 1.554422, 2.091304]

fig1, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(sizes, ratio_1, marker='s', linewidth=2, markersize=6, label='Input of Length 10000000')
ax1.plot(sizes, ratio_2, marker='s', linewidth=2, markersize=6, label='Input of Length 20000000')
ax1.plot(sizes, ratio_3, marker='s', linewidth=2, markersize=6, label='Input of Length 40000000')
ax1.plot(sizes, ratio_4, marker='s', linewidth=2, markersize=6, label='Input of Length 80000000')
ax1.plot(sizes, ratio_5, marker='s', linewidth=2, markersize=6, label='Input of Length 160000000')
ax1.plot(sizes, ratio_6, marker='s', linewidth=2, markersize=6, label='Input of Length 320000000')
ax1.set_xscale('log')
ax1.set_xlabel('Range of Input Elements', fontsize=11)
ax1.set_ylabel('Algorithm Runtime in MS', fontsize=11)
ax1.set_title('MSD vs LSD Radix Sort ratio', fontsize=11)
ax1.legend(frameon=True)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig('../../graphs/iterations3_3.pdf', bbox_inches='tight')