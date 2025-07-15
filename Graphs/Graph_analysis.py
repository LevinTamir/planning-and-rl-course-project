import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
files = ['LearningRate0_0007', 'NewReward']
df1 = pd.read_csv(files[0] + '.csv')
df2 = pd.read_csv(files[1] + '.csv')
# df3 = pd.read_csv(files[2] + '.csv')

# Create the plot
plt.figure(figsize=(12, 6))

# Plot original data
plt.plot(df1['Step'], df1['Value'], alpha=0.7, linewidth=0.8, label='Low yaw cost')
plt.plot(df2['Step'], df2['Value'], alpha=0.7, linewidth=0.8, label='High yaw cost')
# plt.plot(df3['Step'], df3['Value'], alpha=0.7, linewidth=0.8, label='LR=0.0002')

# Apply a simple moving average filter
window_size = 20
filtered_values1 = df1['Value'].rolling(window=window_size, center=True).mean()
filtered_values2 = df2['Value'].rolling(window=window_size, center=True).mean()
# filtered_values3 = df3['Value'].rolling(window=window_size, center=True).mean()

# Plot filtered data
plt.plot(df1['Step'], filtered_values1, linewidth=2, label=f'Low Yaw filtered')
plt.plot(df2['Step'], filtered_values2, linewidth=2, label=f'High Yaw filtered')
# plt.plot(df3['Step'], filtered_values3, linewidth=2, label=f'Filtered Reward (LR={files[2].split("LearningRate")[-1]})')

# Customize the plot
plt.xlabel('Step')
plt.ylabel('Reward Value')
plt.title('DQN Training Rewards - Original vs Filtered')
plt.legend()
plt.grid(True, alpha=0.3)

# Add some statistics
# original_mean = df['Value'].mean()
# filtered_mean = filtered_values.mean()
# plt.axhline(y=original_mean, color='blue', linestyle='--', alpha=0.5, label=f'Original Mean: {original_mean:.2f}')
# plt.axhline(y=filtered_mean, color='red', linestyle='--', alpha=0.5, label=f'Filtered Mean: {filtered_mean:.2f}')

plt.legend()
plt.tight_layout()
plt.show()
