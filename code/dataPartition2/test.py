import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sizes = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000, 1000000000000, 10000000000000, 100000000000000, 1000000000000000, 10000000000000000, 100000000000000000, ]
thrust = [703, 697, 693, 691, 687, 691, 688, 684, 683, 683, 683, 682, 683, 683, 683, 687]
tbb = [205, 191, 183, 176, 171, 169, 171, 171, 168, 169, 170, 171, 169, 169, 169, 171]
ratio = [0.291, 0.274, 0.264, 0.255, 0.249, 0.245, 0.249, 0.250, 0.246, 0.247, 0.249, 0.251, 0.247, 0.248, 0.248, 0.248]

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
plt.savefig('../../graphs/dataPartition2_1.pdf', bbox_inches='tight')


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
plt.savefig('../../graphs/dataPartition2_2.pdf', bbox_inches='tight')


fig3, ax3 = plt.subplots(figsize=(7, 4.5))
ax3.plot(sizes, ratio, marker='o', linewidth=2, markersize=6, label='ratio between CPU sort & GPU sort')
ax3.set_xscale('log')
ax3.set_xlabel('Problem Size (Number of Elements)', fontsize=11)
ax3.set_ylabel('Ratio of Processing Speed', fontsize=11)
ax3.set_title('Ratio of CPU to GPU', fontsize=13, weight='bold')
ax3.legend(frameon=True)
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig('../../graphs/dataPartition2_3.pdf', bbox_inches='tight')


