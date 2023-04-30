import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import datetime

# Load the data from file
with open('log.txt', 'r') as f:
    data_str = f.read().splitlines()

# Convert the data from string to dictionary format
data = []
for d in data_str:
    data.append(eval(d))

max_time = datetime.datetime(2023, 4, 11, 18, 56) #2023-04-11T1856
min_time = datetime.datetime(2023, 4, 8)

# Extract the time and mask size from each data point
times = []
mask_sizes = []
for d in data:
    time_str = d['filename'].split('_')[0]
    time = datetime.datetime.strptime(time_str, '%Y-%m-%dT%H%M%S.%f')
    if min_time < time < max_time:
        # Subtract the minimum time from each timestamp to start from 0
        t = (time - min_time).total_seconds()
        times.append(t)
        mask_sizes.append(d['mask_size'])

# Define the function to fit to the data
def func(x, a, b):
    return a * x + b

# Fit the function to the data
popt, pcov = curve_fit(func, np.array(times), np.array(mask_sizes))

# Create the plot
# Add text annotation for optimized parameters
a_opt = popt[0]
b_opt = popt[1]
plt.text(0, 82000, f"f(x) = a * x + b: {a_opt:.2f} * x + {b_opt:.2f}", fontsize=10)

plt.plot(times, mask_sizes, 'bo', label='Data')
plt.plot(times, func(np.array(times), *popt), 'r-', label='Fit')
plt.xlabel('Time (seconds)')
plt.ylabel('Mask Size')
plt.title('Mask Size vs. Time')
plt.legend()
#plt.savefig('mask_size_plot.png')
plt.show()
