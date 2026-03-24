import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sizes  = [100000, 200000, 300000, 400000, 500000]
tbb    = [771.6,  965.9,  1165.8, 1378.1, 1581.0]

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(sizes, tbb, marker='s', label='CPU (TBB)')

ax.set_xlabel('Problem size (elements)')
ax.set_ylabel('Speed (Melems/s)')
ax.set_title('Sorting speed — Thrust GPU')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

# Save as PDF — vector, perfect quality in LaTeX
plt.tight_layout()
plt.savefig('testingOnParallelCPU.pdf', bbox_inches='tight')