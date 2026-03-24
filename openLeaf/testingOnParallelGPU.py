import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sizes  = [100000, 200000, 300000, 400000, 500000]
gpu    = [1824.6, 2430.7, 2966.0, 3057.2, 3298.3]

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(sizes, gpu, marker='o', label='GPU (Thrust)')

ax.set_xlabel('Problem size (elements)')
ax.set_ylabel('Speed (Melems/s)')
ax.set_title('Sorting speed — TBB CPU')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

# Save as PDF — vector, perfect quality in LaTeX
plt.tight_layout()
plt.savefig('testingOnParallelGPU.pdf', bbox_inches='tight')