import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load data
data = pd.read_csv('result_0430_xception.csv', sep=',')

# Correct the Absolute Relative Error by multiplying by 100
data['Absolute Relative Error'] *= 100

# Group data by class
grouped_data = data.groupby('Class')

# Iterate over each class
for class_name, group in grouped_data:
    print(f"Class: {class_name}")

    # Histogram for visual inspection
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    group['Pred-gt'].hist(bins=20, alpha=0.7, ax=ax[0])
    ax[0].set_title('Histogram of Pred-gt'+f':{class_name}')
    group['Absolute Relative Error'].hist(bins=20, alpha=0.7, ax=ax[1])
    ax[1].set_title('Histogram of Absolute Relative Error'+f':{class_name}')
    plt.show()

    # Estimate normal distribution parameters
    mean_pred_gt, std_pred_gt = group['Pred-gt'].mean(), group['Pred-gt'].std()
    mean_abs_rel_err, std_abs_rel_err = group['Absolute Relative Error'].mean(), group['Absolute Relative Error'].std()
    print(f"Normal Distribution Estimates for Pred-gt: Mean = {mean_pred_gt}, Std Dev = {std_pred_gt}")
    print(f"Normal Distribution Estimates for Absolute Relative Error: Mean = {mean_abs_rel_err}, Std Dev = {std_abs_rel_err}")

    # Q-Q plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    stats.probplot(group['Pred-gt'], dist="norm", plot=ax[0])
    ax[0].set_title('Q-Q Plot of Pred-gt'+f':{class_name}')
    stats.probplot(group['Absolute Relative Error'], dist="norm", plot=ax[1])
    ax[1].set_title('Q-Q Plot of Absolute Relative Error'+f':{class_name}')
    plt.show()
