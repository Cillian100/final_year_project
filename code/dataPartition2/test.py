import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sizes = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000, 1000000000000, 10000000000000, 100000000000000, 1000000000000000, 10000000000000000, 100000000000000000, ]
thrust = [507, 485, 475, 474, 448, 450, 433, 432, 420, 413, 413, 413, 423, 408, 412, 424]
tbb = [171, 157, 148, 145, 142, 141, 140, 146, 145, 147, 141, 145, 149, 144, 140, 141]
fig1, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(sizes, thrust, marker='s', linewidth=2, markersize=6, label='Thrust (GPU)')
ax1.set_xscale('log')
ax1.set_xlabel('Problem Size (Number of Elements)', fontsize=11)
ax1.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)
ax1.set_title('CPU TBB Sorting', fontsize=13, weight='bold')
ax1.legend(frameon=True)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig('dataPartition2_1.pdf', bbox_inches='tight')


fig2, ax2 = plt.subplots(figsize=(7, 4.5))
ax2.plot(sizes, tbb, marker='o', linewidth=2, markersize=6, label='TBB (CPU)')
ax2.set_xscale('log')
ax2.set_xlabel('Problem Size (Number of Elements)', fontsize=11)
ax2.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)
ax2.set_title('GPU Thrust Sorting', fontsize=13, weight='bold')
ax2.legend(frameon=True)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig('dataPartition2_2.pdf', bbox_inches='tight')
