import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sizes = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000, 2048000, 4096000, 8192000, 16384000, 32768000, 65536000, 131072000]
GPU1 = [18, 36, 20, 39, 78, 148, 260, 315, 494, 655, 736, 813, 870, 730, 659, 661, 639, 629]
GPU2 = [18, 36, 20, 39, 78, 148, 261, 314, 494, 654, 738, 814, 873, 731, 658, 636, 637, 628]
CPU = [43, 87, 125, 187, 229, 271, 246, 273, 274, 275, 270, 218, 176, 166, 151, 148, 136, 133]
ratio1 = [2.394635, 2.386190, 6.403000, 4.767321, 2.939580, 1.836893, 0.945901, 0.866559, 0.554552, 0.420630, 0.366750, 0.267706, 0.202305, 0.227632, 0.229619, 0.223873, 0.212738, 0.211572]
ratio2 = [0.998671, 1.003275, 0.990290, 0.995331, 0.994580, 0.996388, 0.996722, 1.003150, 0.999861, 1.000761, 0.998100, 0.998771, 0.996636, 0.998726, 1.001053, 1.038984, 1.002817, 1.001281]
fig1, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(sizes, GPU1, marker='s', linewidth=2, markersize=6, label='Thrust (GPU)')
ax1.set_xscale('log')
ax1.set_xlabel('Problem Size (Number of Elements)', fontsize=11)
ax1.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)
ax1.set_title('CPU TBB Sorting', fontsize=13, weight='bold')
ax1.legend(frameon=True)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig('../graphs/dataPartitionGPU1.pdf', bbox_inches='tight')


fig2, ax2 = plt.subplots(figsize=(7, 4.5))
ax2.plot(sizes, GPU2, marker='s', linewidth=2, markersize=6, label='Thrust (GPU)')
ax2.set_xscale('log')
ax2.set_xlabel('Problem Size (Number of Elements)', fontsize=11)
ax2.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)
ax2.set_title('CPU TBB Sorting', fontsize=13, weight='bold')
ax2.legend(frameon=True)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig('../graphs/dataPartitionGPU1.pdf', bbox_inches='tight')


fig3, ax3 = plt.subplots(figsize=(7, 4.5))
ax3.plot(sizes, CPU, marker='o', linewidth=2, markersize=6, label='TBB (CPU)')
ax3.set_xscale('log')
ax3.set_xlabel('Problem Size (Number of Elements)', fontsize=11)
ax3.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)
ax3.set_title('GPU Thrust Sorting', fontsize=13, weight='bold')
ax3.legend(frameon=True)
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig('../graphs/dataPartitionCPU.pdf', bbox_inches='tight')


fig4, ax4 = plt.subplots(figsize=(7, 4.5))
ax4.plot(sizes, ratio1, marker='o', linewidth=2, markersize=6, label='ratio between CPU sort & GPU sort')
ax4.set_xscale('log')
ax4.set_xlabel('Problem Size (Number of Elements)', fontsize=11)
ax4.set_ylabel('Ratio of Processing Speed', fontsize=11)
ax4.set_title('Ratio of CPU to GPU', fontsize=13, weight='bold')
ax4.legend(frameon=True)
ax4.grid(True, linestyle='--', alpha=0.6)
ax4.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig('../graphs/dataPartitionRatioCPUandGPU.pdf', bbox_inches='tight')


fig5, ax5 = plt.subplots(figsize=(7, 4.5))
ax5.plot(sizes, ratio2, marker='o', linewidth=2, markersize=6, label='ratio between GPU1 and GPU2')
ax5.set_xscale('log')
ax5.set_xlabel('Problem Size (Number of Elements)', fontsize=11)
ax5.set_ylabel('Ratio of Processing Speed', fontsize=11)
ax5.set_title('Ratio of GPU1 to GPU2', fontsize=13, weight='bold')
ax5.legend(frameon=True)
ax5.grid(True, linestyle='--', alpha=0.6)
ax5.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig('../graphs/dataPartitionRatioGPU1andGPU2.pdf', bbox_inchest='tight')
