import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sizes = [1000000, 2000000, 4000000, 8000000, 16000000, 32000000, 64000000, 128000000, 256000000, 512000000]
GPU1 = [208.339592, 240.190100, 285.364356, 330.680059, 375.961727, 374.787097, 377.094930, 377.822678, 378.641540, 378.556734]
GPU2 = [257.057704, 296.610505, 357.425739, 380.123557, 368.399164, 372.628327, 382.070476, 383.633356, 384.366695, 384.384166]
CPU1 = [36.449936, 69.493513, 118.024816, 141.473546, 261.532638, 328.702253, 466.697027, 580.785208, 678.535736, 728.121858]
CPU2 = [128.158616, 185.352554, 207.584900, 264.009079, 319.541039, 311.869692, 314.082001, 337.248452, 342.721556, 355.805462]
ratio1 = [0.174954, 0.289327, 0.413593, 0.427826, 0.695636, 0.877037, 1.237612, 1.537190, 1.792027, 1.923415]
ratio2 = [0.810478, 0.809783, 0.798388, 0.869928, 1.020528, 1.005793, 0.986977, 0.984854, 0.985105, 0.984840]

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
plt.savefig('../../graphs/dataPartitionNew/dataPartitionGPU1.pdf', bbox_inches='tight')

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
plt.savefig('../../graphs/dataPartitionNew/dataPartitionGPU2.pdf', bbox_inches='tight')

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
plt.savefig('../../graphs/dataPartitionNew/dataPartitionCPU1.pdf', bbox_inches='tight')

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
plt.savefig('../../graphs/dataPartitionNew/dataPartitionCPU2.pdf', bbox_inches='tight')

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
plt.savefig('../../graphs/dataPartitionNew/dataPartitionRatioCPUandGPU.pdf', bbox_inches='tight')

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
plt.savefig('../../graphs/dataPartitionNew/dataPartitionRatioGPU1andGPU2.pdf', bbox_inches='tight')
