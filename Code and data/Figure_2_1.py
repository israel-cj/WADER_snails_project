import json
import matplotlib.pyplot as plt
import pandas as pd
import re
from matplotlib import rc

# activate latex text rendering
rc('text', usetex=True)


# Function to generate a group key
def get_group_key(key):
    return re.sub(r'\.\d+', '', key)

# Read the data from the JSON files
with open('countr.json', 'r') as f:
    countr_data = json.load(f)
with open('deep_data.json', 'r') as f:
    deepdata_data = json.load(f)
with open('grounded_counting.json', 'r') as f:
    grounded_countingDINO_data = json.load(f)
with open('SAM_count_anything.json', 'r') as f:
    SAM_count_anything_data = json.load(f)
with open('free_training.json', 'r') as f:
    free_training = json.load(f)
with open('real.json', 'r') as f:
    real_data = json.load(f)
with open('API_deepdata.json', 'r') as f:
    API_deep = json.load(f)

# Create dataframes from the dictionaries
df_countr = pd.DataFrame(list(countr_data.items()), columns=['key', 'countr'])
df_deepdata = pd.DataFrame(list(deepdata_data.items()), columns=['key', 'deepdataspace'])
df_grounded_countingDINO = pd.DataFrame(list(grounded_countingDINO_data.items()), columns=['key', 'grounded_countingDINO'])
df_SAM_count_anything = pd.DataFrame(list(SAM_count_anything_data.items()), columns=['key', 'SAM'])
df_free_training = pd.DataFrame(list(free_training.items()), columns=['key', 'training_free_obj'])
df_real = pd.DataFrame(list(real_data.items()), columns=['key', 'real'])
# df_API_deep = pd.DataFrame(list(API_deep.items()), columns=['key', 'api_deepdata_space'])

# Merge the dataframes on the 'key' column
df = pd.merge(df_countr, df_deepdata, on='key')
df = pd.merge(df, df_grounded_countingDINO, on='key')
df = pd.merge(df, df_SAM_count_anything, on='key')
df = pd.merge(df, df_free_training, on='key')
df = pd.merge(df, df_real, on='key')
#df = pd.merge(df, df_API_deep, on='key')

# Generate group keys and group by them, summing the values
df['group_key'] = df['key'].apply(get_group_key)
df_grouped = df.groupby('group_key').sum()


print(df_grouped.columns)

df_grouped = df_grouped.sort_index()
# I want to change the names of the next columns by adding a "*" in the end, indicating the statistical difference [countr, grounded_countingDINO, segment_anything, training_free_obj]
df_grouped = df_grouped.rename(columns={'countr': 'CounTR', 'grounded_countingDINO': 'Grounding DINO', 'training_free_obj': 'Training-free Object Counting', 'deepdataspace': 'DeepDataSpace'})

print('df_grouped columns', df_grouped.columns)
#######


import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm

# Assuming df_grouped and columns_extra are already defined
columns_extra = ['DeepDataSpace', 'Training-free Object Counting', 'SAM', 'Grounding DINO', 'CounTR']
names_x_axis = list(df_grouped.index)
new_x_sticks = []
for i in range(9):
    new_x_sticks.append(str((i+1)*100)+' C')
    new_x_sticks.append(str((i+1)*100)+' S')

# Filter numeric columns
numeric_df_grouped = df_grouped.select_dtypes(include=[float, int])

# Calculate variance for numeric columns
variance_values = numeric_df_grouped.var()
std_values = variance_values ** 0.5

# Generate colors from the viridis colormap
viridis = cm.get_cmap('viridis', len(columns_extra))
colors = viridis(range(len(columns_extra)))

# Create a dictionary mapping columns to colors
color_dict = {column: colors[idx] for idx, column in enumerate(columns_extra)}

print("color_dict", color_dict)

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the data
for column in columns_extra:
    if column in numeric_df_grouped.columns:
        ax.plot(df_grouped.index, df_grouped[column], label=column, color=color_dict[column])

        # Add shaded areas for the variance
        ax.fill_between(df_grouped.index, df_grouped[column] - std_values[column], df_grouped[column] + std_values[column], color=color_dict[column], alpha=0.2)

# Add vertical lines for each x-axis separation
for x in range(len(names_x_axis)):
    ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)

# Set the new x-axis labels to new_x_sticks
ax.set_xticks(ticks=range(len(new_x_sticks)))
ax.set_xticklabels(labels=new_x_sticks)

# Set the xlabel with italic 'P. ulvae' and 'Density & Aggregation'
ax.set_xlabel(r'\textit{P. ulvae} Density \& Aggregation', fontweight='bold')

# Set the ylabel with italic 'P. ulvae' and 'count'
ax.set_ylabel(r'\textit{P. ulvae} count', fontweight='bold')

# CounTR, SAM, Training Free object counting, Grounding DINO, DeepDataSpace.
# Desired legend order
legend_order = ['CounTR', 'SAM', 'Training-free Object Counting', 'Grounding DINO', 'DeepDataSpace']

# Get handles and labels from the current plot
handles, labels = ax.get_legend_handles_labels()
print("labels", labels)

# Create a dictionary to map labels to handles
label_to_handle = dict(zip(labels, handles))

# Reorder handles and labels according to the desired legend order
ordered_handles = [label_to_handle[label] for label in legend_order if label in label_to_handle]
ordered_labels = [label for label in legend_order if label in label_to_handle]

# Add legend with the specified order
ax.legend(ordered_handles, ordered_labels, loc='upper left')

# Adjust layout to not cut off labels
fig.tight_layout()

# Save the figure
fig.savefig('carrying_capacity_improved.png', bbox_inches='tight')
fig.savefig('Figure_3.eps', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

####

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison

# Assuming df_grouped is your DataFrame
# First, melt the DataFrame to long format
df_long = pd.melt(df_grouped, id_vars=['key'], var_name='device', value_name='count')

# Categorize counts as 'device' or 'human'
df_long['count_type'] = df_long['device'].apply(lambda x: 'human' if x == 'real' else 'device')

# Two-way ANOVA
model = ols('count ~ C(device) + C(count_type) + C(device):C(count_type)', data=df_long).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
print("Anova results")
print(anova_results)

# Post Hoc Analysis if needed
# Adjust the groupings as necessary based on your specific categories
mc = MultiComparison(df_long['count'], df_long['device'])
tukey_result = mc.tukeyhsd()
print("Tukey results")
print(tukey_result.summary())

# Save the summary as TXT 
with open('anova_results.txt', 'w') as f:
    f.write(str(anova_results))
    f.write('\n\n')
    f.write(str(tukey_result.summary()))    