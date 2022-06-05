import matplotlib.pyplot as plt
from matplotlib import style
from read_dataset import df
import numpy as np


# ========================= Plot the Data  =================================
colors = ['b', 'r', 'g', 'k', 'm']
style.use('ggplot')
all_groups = df.groupby('genre')

# ===== Scatter HipHop ========
hip = all_groups.get_group(0)
hip_np = np.array(hip)
plt.scatter(hip_np[:, 0], hip_np[:, 1], c=colors[0], marker='x')
plt.title('HipHop')
plt.xlabel("Age")
plt.ylabel("Gender")
plt.show()

# ===== Scatter Jazz ========
jazz = all_groups.get_group(1)
jazz_np = np.array(jazz)
plt.scatter(jazz_np[:, 0], jazz_np[:, 1], c=colors[1], marker='x')
plt.title('Jazz')
plt.xlabel("Age")
plt.ylabel("Gender")
plt.show()

# ===== Scatter Classical ========
classical = all_groups.get_group(2)
classical_np = np.array(classical)
plt.scatter(classical_np[:, 0], classical_np[:, 1], c=colors[2], marker='x')
plt.title('Classical')
plt.xlabel("Age")
plt.ylabel("Gender")
plt.show()


# ===== Scatter Dance ========
dance = all_groups.get_group(3)
dance_np = np.array(dance)
plt.scatter(dance_np[:, 0], dance_np[:, 1], c=colors[3], marker='x')
plt.title('Dance')
plt.xlabel("Age")
plt.ylabel("Gender")
plt.show()

# ===== Scatter Acoustic ========
acoustic = all_groups.get_group(4)
acoustic_np = np.array(acoustic)
plt.scatter(acoustic_np[:, 0], acoustic_np[:, 1], c=colors[4], marker='x')
plt.title('Acoustic')
plt.xlabel("Age")
plt.ylabel("Gender")
plt.show()

# ============ PLot all data in one =========

plt.scatter(acoustic_np[:, 0], acoustic_np[:, 1], c=colors[4], marker='x')
plt.plot([], [], color=colors[4], label='acoustic')

plt.scatter(dance_np[:, 0], dance_np[:, 1], c=colors[3], marker='x')
plt.plot([], [], color=colors[3], label='Dance')

plt.scatter(classical_np[:, 0], classical_np[:, 1], c=colors[2], marker='x')
plt.plot([], [], color=colors[2], label='Classical')

plt.scatter(jazz_np[:, 0], jazz_np[:, 1], c=colors[1], marker='o')
plt.plot([], [], color=colors[1], label='Jazz')

plt.scatter(hip_np[:, 0], hip_np[:, 1], c=colors[0], marker='x')
plt.plot([], [], color=colors[0], label='HipHop')

plt.xlabel('Age')
plt.ylabel("Gender")
plt.legend()
plt.show()
