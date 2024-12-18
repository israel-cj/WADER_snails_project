import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import mannwhitneyu
from matplotlib import rc

# activate latex text rendering
rc('text', usetex=True)

real_df = pd.read_csv('Count humans.csv')
# Calculate mean and std for real_df
real_df['mean'] = real_df.loc[:, real_df.columns != 'name'].mean(axis=1).astype(int)
real_df['std'] = real_df.loc[:, real_df.columns != 'name'].std(axis=1)


# Delete '.JPG' from the 'name' column
real_df['name'] = real_df['name'].str.replace('.JPG', '')

counter_df = pd.read_csv('CounTR.csv')
DeepData_df = pd.read_csv('DeepData.csv')
Grounded_df = pd.read_csv('Grounded.csv')
SAM_df = pd.read_csv('SAM.csv')
TrainingFree_df = pd.read_csv('TrainingFree.csv')

list_df =[counter_df, DeepData_df, Grounded_df, SAM_df, TrainingFree_df]
# Sort the 'name' column in each dataframe and put them into a new list 'list_df_sorted'
list_df_sorted = []
for df in list_df:
    df['name'] = df['name'].str.replace('.JPG', '')
    df = df[['name', 'prediction_1', 'prediction_2', 'prediction_3', 'prediction_4', 'prediction_5', 'prediction_6']]  # Select only the required columns
    df_sorted = df.sort_values('name')
    list_df_sorted.append(df_sorted)

# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
colors = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725', '#440154', '#3b528b']


# Add a new column 'mean' and 'std' to each DataFrame in 'list_df_sorted' that is the mean and standard deviation of all other columns
for df in list_df_sorted:
    df['mean'] = df.loc[:, df.columns != 'name'].mean(axis=1)
    df['std'] = df.loc[:, df.columns != 'name'].std(axis=1)

# Statiscal difference:
# list_df_names =['CounTR', 'DeepDataSpace', 'Grounded counting Dino', 'SAM*', 'Training free object*']
list_df_names = ['CounTR', 'DeepDataSpace', 'Grounding DINO', 'SAM', 'Training-free Object Counting']

color_dict = {
    'DeepDataSpace': [0.267004, 0.004874, 0.329415, 1.],
    'Training-free Object Counting': [0.229739, 0.322361, 0.545706, 1.],
    'SAM': [0.127568, 0.566949, 0.550556, 1.],
    'Grounding DINO': [0.369214, 0.788888, 0.382914, 1.],
    'CounTR': [0.993248, 0.906157, 0.143936, 1.]
}

new_x_sticks = []
for i in range(4):
    new_x_sticks.append(str((i+1)*100)+' C')
    new_x_sticks.append(str((i+1)*100)+' S')

plt.figure(figsize=(12, 8))
# Plot 'name' on the x-axis and 'mean' on the y-axis for each DataFrame
for i, (df, name) in enumerate(zip(list_df_sorted, list_df_names)):
    mse = mean_squared_error(real_df['mean'], df['mean'])
    # Perform Mann-Whitney U test
    mwu_result = mannwhitneyu(real_df['mean'], df['mean'])
    plt.plot(df['name'], df['mean'], label=f'{name}', color=color_dict[name])
    plt.fill_between(df['name'], df['mean'] - df['std'], df['mean'] + df['std'], color=color_dict[name], alpha=0.4)

plt.xlabel(r'\textit{P. ulvae} Density \& Aggregation')
plt.ylabel(r'\textit{P. ulvae} count')
plt.plot(real_df['mean'], color='gray', linewidth=3, label='Real')
plt.fill_between(real_df.index, real_df['mean'] - real_df['std'], real_df['mean'] + real_df['std'], color='gray', alpha=0.4)
# I need the legens in the next order: DeepDataSpace, Training-free Object Counting, SAM, Grounding DINO, CounTR

# Assuming you have already created the plot and added the legend
handles, labels = plt.gca().get_legend_handles_labels()


# Define the desired order
# order = ['DeepDataSpace', 'Training-free Object Counting', 'SAM', 'Grounding DINO', 'CounTR']
order = ['CounTR', 'SAM', 'Training-free Object Counting', 'Grounding DINO', 'DeepDataSpace', 'Real']


# Reorder the handles and labels
ordered_handles = [handles[labels.index(label)] for label in order]
ordered_labels = [labels[labels.index(label)] for label in order]

# I want to change Real for Human in the legend
ordered_labels[-1] = 'Human'

# Set the legend with the reordered handles and labels
plt.legend(ordered_handles, ordered_labels, loc='upper left')

# plt.legend(bbox_to_anchor=(1.3, 0.5), loc='upper left')
plt.xticks(ticks=range(len(new_x_sticks)), labels=new_x_sticks)
plt.savefig('together_MSE_statistical.png', bbox_inches='tight')
plt.savefig('Figure_4.eps', dpi=300, bbox_inches='tight')
plt.close()

# clean image
plt.clf()
plt.figure(figsize=(12, 8))
# Plot 'name' on the x-axis and 'mean' on the y-axis for each DataFrame
for i, (df, name) in enumerate(zip(list_df_sorted, list_df_names)):
    r2 = r2_score(real_df['mean'], df['mean'])
    # Perform Mann-Whitney U test
    mwu_result = mannwhitneyu(real_df['mean'], df['mean'])
    plt.plot(df['name'], df['mean'], label=f'{name}', color=colors[i % len(colors)])
    plt.fill_between(df['name'], df['mean'] - df['std'], df['mean'] + df['std'], color=colors[i % len(colors)], alpha=0.4)
plt.xlabel('Testing images', fontweight='bold')
plt.ylabel('Marine macroinvertebrate', fontweight='bold')
# plt.title('Performance of the models over 6 counting iterations with exemplars variance')
plt.plot(real_df['mean'], color='gray', linewidth=3, label='Real')  
plt.fill_between(real_df.index, real_df['mean'] - real_df['std'], real_df['mean'] + real_df['std'], color='blueviolet', alpha=0.4)
plt.legend(bbox_to_anchor=(1.3, 0.5), loc='upper center')
plt.xticks(rotation=25, fontsize=8)
plt.savefig('together_R2_statistical.png', bbox_inches='tight')  # Save the figure as a .png file
plt.close()  # Close the figure without displaying it