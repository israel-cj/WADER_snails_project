import json
import pandas as pd
import re
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

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

# Reshape the DataFrame from wide to long format
df_long = pd.melt(df, id_vars=['key'], value_vars=['countr', 'deepdataspace', 'grounded_countingDINO', 'SAM', 'training_free_obj', 'real'],
                  var_name='models', value_name='count')

print("df_long")
print(df_long)
df_long.to_csv("Figure_3.csv", index=False)

df_long_separted = df_long.copy()
# Create a new column 'category' based on the 'key' column
df_long_separted['category'] = df_long_separted['key'].apply(lambda x: 'Clumped' if 'Clumped' in x else 'Spaced')
df_long_separted.to_csv("Figure_3_Clumped_Spaced.csv", index=False)


# Perform ANOVA
model = ols('count ~ C(models)', data=df_long).fit()
anova_results = anova_lm(model)
print(anova_results)

# Perform Tukey's HSD post-hoc test
tukey = pairwise_tukeyhsd(endog=df_long['count'], groups=df_long['models'], alpha=0.05)
print(tukey)

# Plot the results of Tukey's HSD
tukey.plot_simultaneous()
plt.show()