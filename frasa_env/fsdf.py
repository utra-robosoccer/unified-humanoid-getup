import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the CSV file, skipping lines starting with '#'
monitor_df = pd.read_csv('/home/manx52/catkin_ws/src/frasa/frasa_env/0.monitor.csv', comment='#')

# Step 2: Inspect the data
print("Data from monitor.csv:")
print(monitor_df.head())

# Step 3: Compute some basic statistics (e.g., average reward)
average_reward = monitor_df['r'].mean()
print("Average Reward:", average_reward)

# Step 4: Plot the rewards per episode
plt.figure(figsize=(10, 5))
plt.plot(monitor_df['t'], monitor_df['r'], label='Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Rewards over Time')
plt.legend()
plt.xlim(0, 2500)

plt.grid(True)
plt.show()
