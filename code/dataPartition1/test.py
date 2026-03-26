import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sizes = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000, 2048000, 4096000, 8192000, 16384000, 32768000, 65536000, 131072000, 262144000, 524288000]
thrust = [4, 9, 4, 9, 18, 36, 69, 110, 177, 230, 275, 307, 388, 456, 464, 476, 482, 482, 482, 482]
tbb = [1, 7, 1, 8, 24, 34, 56, 82, 136, 185, 166, 150, 200, 170, 146, 149, 137, 136, 127, 132]
fig, ax = plt.subplots(figsize=(7, 4.5))

# Plot data with clearer styling
ax.plot(sizes, thrust, marker='s', linewidth=2, markersize=6, label='Thrust (GPU)')
ax.plot(sizes, tbb, marker='o', linewidth=2, markersize=6, label='TBB (CPU)')
ax.set_xscale('log')

# Labels and title
ax.set_xlabel('Problem Size (Number of Elements)', fontsize=11)
ax.set_ylabel('Processing Speed (Million Elements per Second)', fontsize=11)
ax.set_title('Sorting Performance Comparison: GPU vs CPU', fontsize=13, weight='bold')

# Improve legend and grid
ax.legend(frameon=True)
ax.grid(True, linestyle='--', alpha=0.6)

# Optional: format ticks for readability
ax.tick_params(axis='both', labelsize=10)

# Tight layout and save
plt.tight_layout()
plt.savefig('sorting_performance.pdf', bbox_inches='tight')
