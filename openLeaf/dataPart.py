import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sizes = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000, 2048000, 4096000, 8192000, 16384000, 32768000]
tbb = [11, 117, 98, 41, 201, 167, 396, 386, 374, 373, 466, 434, 376, 369, 390, 448]
thrust = [28, 116, 89, 282, 655, 1425, 2153, 3234, 3726, 4460, 5286, 5560, 5468, 5535, 5550, 5548]


fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(sizes, thrust, marker='s', label='thrust')
ax.plot(sizes, tbb, marker='o', label='tbb')

ax.set_xlabel('Problem size (elements)')
ax.set_ylabel('Speed (Melems/s)')
ax.set_title('Sorting speed — Thrust GPU')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

# Save as PDF — vector, perfect quality in LaTeX
plt.tight_layout()
plt.savefig('poop.pdf', bbox_inches='tight')
