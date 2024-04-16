
import pandas as pd
import scienceplots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import plotly.express as px

# Load your data from the CSV file
df = pd.read_csv('benchmark_results.csv')

# Extract the columns of interest
memory = df['Memory (MiB)']
vcpus = df['vCPUs']
execution_time = df['Avg. Execution Time']

# 2D Scatter Plot (Memory vs. Execution Time)
plt.style.use(['science', 'ieee', 'no-latex'])
plt.figure(figsize=(8, 6))  # Adjust figure size as needed 
plt.grid(True, 'both', 'both')
plt.scatter(memory, vcpus, c=execution_time, cmap='viridis')  # Color by executiontime
plt.xlabel('Memory (MiB)')
plt.ylabel('vCPUs')
plt.xscale('log')
plt.yscale('log')

plt.title('Execution time \n by Memory and vCPUs')
plt.colorbar(label='Avg. Execution Time')
plt.show()


# 3D plot using matplotlib
# plt.style.use(['science', 'ieee', 'no-latex'])
# fig = plt.figure(figsize=(10, 8))  
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(memory, vcpus, execution_time, c=execution_time, cmap='coolwarm')
# ax.set_xlabel('Memory (MiB)')
# ax.set_ylabel('vCPUs')
# ax.set_zlabel('Avg. Execution Time')
# plt.title('3D Scatter Plot')
# plt.show()

# fig = px.scatter_3d(df, x='Memory (MiB)', y='vCPUs', z='Avg. Execution Time',
#                     color='Avg. Execution Time', size='Avg. Execution Time')
# fig.update_layout(title='Plotly 3D Scatter Plot')
# fig.show()
