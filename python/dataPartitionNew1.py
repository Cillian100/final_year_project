import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sizes = [10000, 20000, 40000, 80000, 160000, 320000, 640000, 1280000, 2560000, 5120000, 10240000, 20480000, 40960000, 81920000, 163840000, 327680000, 655360000]
GPU1 = [10.838536, 22.217476, 43.562979, 83.249687, 133.919595, 186.561714, 239.377500, 278.461345, 326.748329, 374.042393, 379.529808, 370.019964, 375.509511, 383.336866, 384.696950, 385.230522, 385.464779]
GPU2 = [11.127868, 22.819618, 44.863931, 85.463026, 137.608692, 191.055387, 244.782538, 284.510140, 333.813267, 379.425726, 385.379954, 389.354666, 388.115126, 389.232063, 390.506001, 390.423087, 389.702972]
CPU1 = [0.227610, 0.476403, 0.707936, 1.209254, 2.347665, 4.581985, 10.046090, 18.556388, 32.233944, 61.085404, 118.983813, 184.304995, 264.778695, 331.239901, 381.340317, 369.586220, 367.250095]
CPU2 = [35.760671, 110.319620, 122.304300, 147.124658, 138.477852, 162.826265, 175.585618, 196.197347, 207.968895, 207.600324, 229.777387, 318.081854, 302.066459, 353.582032, 352.035886, 345.089214, 353.393353]
ratio1 = [0.021000, 0.021443, 0.016251, 0.014526, 0.017530, 0.024560, 0.041968, 0.066639, 0.098651, 0.163311, 0.313503, 0.498095, 0.705118, 0.864096, 0.991275, 0.959390, 0.952746]
ratio2 = [0.973999, 0.973613, 0.971002, 0.974102, 0.973191, 0.976480, 0.977919, 0.978740, 0.978836, 0.985812, 0.984820, 0.950342, 0.967521, 0.984854, 0.985124, 0.986700, 0.989125]

fig1, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(sizes, GPU1, marker='s', linewidth=2, markersize=6, label='Thrust GPU1')
ax1.set_xscale('log')
ax1.set_xlabel('Problem Size (Number of Elements)', fontsize=11)
ax1.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)
ax1.set_title('GPU1 Thrust Sorting', fontsize=13, weight='bold')
ax1.legend(frameon=True)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig('../graphs/dataPartitionNew/dataPartitionGPU1.pdf', bbox_inches='tight')

fig2, ax2 = plt.subplots(figsize=(7, 4.5))
ax2.plot(sizes, GPU2, marker='s', linewidth=2, markersize=6, label='Thrust GPU2')
ax2.set_xscale('log')
ax2.set_xlabel('Problem Size (Number of Elements)', fontsize=11)
ax2.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)
ax2.set_title('GPU2 Thrust Sorting', fontsize=13, weight='bold')
ax2.legend(frameon=True)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig('../graphs/dataPartitionNew/dataPartitionGPU2.pdf', bbox_inches='tight')

fig3, ax3 = plt.subplots(figsize=(7, 4.5))
ax3.plot(sizes, CPU1, marker='o', linewidth=2, markersize=6, label='RADULS (CPU)')
ax3.set_xscale('log')
ax3.set_xlabel('Problem Size (Number of Elements)', fontsize=11)
ax3.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)
ax3.set_title('CPU RADULS Sorting', fontsize=13, weight='bold')
ax3.legend(frameon=True)
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig('../graphs/dataPartitionNew/dataPartitionCPU1.pdf', bbox_inches='tight')

fig4, ax4 = plt.subplots(figsize=(7, 4.5))
ax4.plot(sizes, CPU2, marker='o', linewidth=2, markersize=6, label='Intel TBB (CPU)')
ax4.set_xscale('log')
ax4.set_xlabel('Problem Size (Number of Elements)', fontsize=11)
ax4.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)
ax4.set_title('CPU Intel TBB Sorting', fontsize=13, weight='bold')
ax4.legend(frameon=True)
ax4.grid(True, linestyle='--', alpha=0.6)
ax4.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig('../graphs/dataPartitionNew/dataPartitionCPU2.pdf', bbox_inches='tight')

fig5, ax5 = plt.subplots(figsize=(7, 4.5))
ax5.plot(sizes, ratio1, marker='o', linewidth=2, markersize=6, label='ratio CPU1 / GPU1')
ax5.set_xscale('log')
ax5.set_xlabel('Problem Size (Number of Elements)', fontsize=11)
ax5.set_ylabel('Ratio of Processing Speed', fontsize=11)
ax5.set_title('Ratio of CPU to GPU', fontsize=13, weight='bold')
ax5.legend(frameon=True)
ax5.grid(True, linestyle='--', alpha=0.6)
ax5.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig('../graphs/dataPartitionNew/dataPartitionRatioCPUandGPU.pdf', bbox_inches='tight')

fig6, ax6 = plt.subplots(figsize=(7, 4.5))
ax6.plot(sizes, ratio2, marker='o', linewidth=2, markersize=6, label='ratio GPU1 / GPU2')
ax6.set_xscale('log')
ax6.set_xlabel('Problem Size (Number of Elements)', fontsize=11)
ax6.set_ylabel('Ratio of Processing Speed', fontsize=11)
ax6.set_title('Ratio of GPU1 to GPU2', fontsize=13, weight='bold')
ax6.legend(frameon=True)
ax6.grid(True, linestyle='--', alpha=0.6)
ax6.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig('../graphs/dataPartitionNew/dataPartitionRatioGPU1andGPU2.pdf', bbox_inches='tight')
