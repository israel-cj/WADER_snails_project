import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


real_df = pd.read_csv('Count humans.csv')
# Calculate mean and std for real_df
real_df['mean'] = real_df.loc[:, real_df.columns != 'name'].mean(axis=1).astype(int)
real_df['std'] = real_df.loc[:, real_df.columns != 'name'].std(axis=1)

counter_df = pd.read_csv('CounTR.csv')
DeepData_df = pd.read_csv('DeepData.csv')
Grounded_df = pd.read_csv('Grounded.csv')
SAM_df = pd.read_csv('SAM.csv')
TrainingFree_df = pd.read_csv('TrainingFree.csv')

list_df =[counter_df, DeepData_df, Grounded_df, SAM_df, TrainingFree_df]
list_df_names =['CounTR', 'DeepDataSpace', 'Grounded counting Dino', 'Segment anything', 'Training free object']

# Sort the 'name' column in each dataframe and put them into a new list 'list_df_sorted'

list_df_sorted = []
for df in list_df:
    df = df[['name', 'prediction_1', 'prediction_2', 'prediction_3', 'prediction_4', 'prediction_5', 'prediction_6']]  # Select only the required columns
    df_sorted = df.sort_values('name')
    list_df_sorted.append(df_sorted)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Add a new column 'mean' and 'std' to each DataFrame in 'list_df_sorted' that is the mean and standard deviation of all other columns
for df in list_df_sorted:
    df['mean'] = df.loc[:, df.columns != 'name'].mean(axis=1)
    df['std'] = df.loc[:, df.columns != 'name'].std(axis=1)

only_mean_list = [df['mean'] for df in list_df_sorted]

# join only the mean per device and the real_df['mean'] in a single dataframe
df = pd.concat([real_df['mean']] + only_mean_list, axis=1)
df.columns = ['Real', 'CounTR', 'DeepData', 'Grounded', 'SAM', 'TrainingFree']
print(df)


import pandas as pd
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Assuming 'df' is your DataFrame

# Step 3: Compute differences (for illustration, this step is conceptual and integrated into the ANOVA and posthoc steps)

# Step 4: Perform ANOVA
# Reshape the DataFrame for ANOVA
df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=['Real', 'CounTR', 'DeepData', 'Grounded', 'SAM', 'TrainingFree'])
df_melt.columns = ['index', 'models', 'value']
print("df_melt")
print(df_melt)

df_melt.to_csv("Figure_4.csv", index=False)

# Perform the ANOVA
anova = ols('value ~ C(models)', data=df_melt).fit()
anova_table = sm.stats.anova_lm(anova, typ=2)
print("ANOVA results")
print(anova_table)	

# Check if the p-value in anova_table is less than 0.05 to proceed with posthoc analysis

# Step 5: Posthoc Analysis with Tukey's HSD
tukey = pairwise_tukeyhsd(endog=df_melt['value'], groups=df_melt['models'], alpha=0.05)
print(tukey.summary())
print(tukey)

# Save the information from tukey into TXT
with open('tukey.txt', 'w') as f:
    # save anova
    f.write(str(anova_table))
    f.write(str(tukey))

# Step 6: Interpret Results
# The printout from Tukey's HSD test will show which method means are significantly different from each other.