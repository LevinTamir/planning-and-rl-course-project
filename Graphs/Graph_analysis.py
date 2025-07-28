import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
# Load the data
dir = os.path.dirname(os.path.abspath(__file__))
files = ['[07m19d-22_26] dqn_stage2_reward', 'Test0_0007']
df1 = pd.read_csv(os.path.join(dir, files[0] + '.csv')) # Learning
df2 = pd.read_csv(os.path.join(dir, files[1] + '.csv')) # Testing
# df3 = pd.read_csv(os.path.join(dir, files[2] + '.csv'))

# Starting df2 step count to start from the end of df1
# df2['Step'] += df1['Step'].max() + 1

# Create the plot
plt.figure(figsize=(12, 6))

# Plot original data
plt.plot(df1['Step'], df1['Value'], alpha=0.7, linewidth=0.8, label='Learning stage')
plt.plot(df2['Step'], df2['Value'], alpha=0.7, linewidth=0.8, label='Testing stage')
# plt.plot(df3['Step'], df3['Value'], alpha=0.7, linewidth=0.8, label='LR=0.0002')

# Apply a simple moving average filter
window_size = 20
filtered_values1 = df1['Value'].rolling(window=window_size, center=True).mean()
filtered_values2 = df2['Value'].rolling(window=window_size, center=True).mean()
# filtered_values3 = df3['Value'].rolling(window=window_size, center=True).mean()

# Plot filtered data
plt.plot(df1['Step'], filtered_values1, linewidth=2, label=f'Learning stage filtered')
plt.plot(df2['Step'], filtered_values2, linewidth=2, label=f'Testing stage filtered')
# plt.plot(df3['Step'], filtered_values3, linewidth=2, label=f'Filtered Reward (LR={files[2].split("LearningRate")[-1]})')

# Customize the plot
plt.xlabel('Step')
plt.ylabel('Reward Value')
# plt.title('DQN Training Rewards - Learning vs Test')
plt.legend()
plt.grid(True, alpha=0.3)

plt.legend()
plt.tight_layout()
plt.show()
