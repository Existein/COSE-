import pandas as pd
from sklearn.linear_model import LinearRegression

# Load your data into a DataFrame
df = pd.read_csv('benchmark_results.csv')

# Model input
X = df[['Memory (MiB)', 'vCPUs']] 
# Target variable
y = df['Avg. Execution Time'] 

model = LinearRegression()
model.fit(X, y)  

# Make predictions on new configurations
new_memory = 768 
new_vcpu = 1.5 
predicted_time = model.predict([[new_memory, new_vcpu]])[0]
print(predicted_time)
