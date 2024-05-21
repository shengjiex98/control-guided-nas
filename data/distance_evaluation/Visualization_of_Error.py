# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:12:27 2024
@author: HuLab
"""

import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the input file
input_file_name = 'result_0430_xception.csv'

# Read the file into a DataFrame
data = pd.read_csv(input_file_name, sep=',', usecols=[1, 5, 6])

# Group the data by 'Class' and calculate the mean and variance of the Pred-gts
grouped_data = data.groupby('Class').agg({'Pred-gt': ['mean', 'var'], 'Absolute Relative Error': ['mean', 'var']})

# Print the mean Pred-gt and mean Absolute Relative Error for each class
print("Mean Pred-gt and Mean Absolute Relative Error for each class:")
print(grouped_data)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Mean Pred-gt plot
grouped_data['Pred-gt']['mean'].plot(kind='bar', color='skyblue', ax=axs[0, 0])
axs[0, 0].set_title('Mean Pred-gt by Class')
axs[0, 0].set_ylabel('Mean Pred-gt')
axs[0, 0].set_xlabel('Class')

# Mean Absolute Relative Error plot
(grouped_data['Absolute Relative Error']['mean']*100).plot(kind='bar', color='salmon', ax=axs[1, 0])
axs[1, 0].set_title('Mean Absolute Relative Error by Class')
axs[1, 0].set_ylabel('Mean Absolute Relative Error')
axs[1, 0].set_xlabel('Class')

# Variance Pred-gt plot
grouped_data['Pred-gt']['var'].plot(kind='bar', color='skyblue', ax=axs[0, 1])
axs[0, 1].set_title('Variance of Pred-gt by Class')
axs[0, 1].set_ylabel('Variance Pred-gt')
axs[0, 1].set_xlabel('Class')

# Variance Absolute Relative Error plot
(grouped_data['Absolute Relative Error']['var'] * 100).plot(kind='bar', color='salmon', ax=axs[1, 1])
axs[1, 1].set_title('Variance of Absolute Relative Error by Class')
axs[1, 1].set_ylabel('Variance Absolute Relative Error')
axs[1, 1].set_xlabel('Class')

# Improve layout and show the plot
plt.tight_layout()
plt.show()
