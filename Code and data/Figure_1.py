import pandas as pd
import numpy as np
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

all_together = {
    'DSLR angle 1': [311, 397, 854, 160, 208, 86, 31, 110, 87, 214],
    'DSLR angle 2': [265, 401, 289, 170, 280, 543, 25, 131, 89, 167],
    'DSLR angle 3': [368, 458, 866, 154, 218, 549, 16, 109, 61, 192],
    'GoPro angle 1': [230, 278, 491, 141, 189, 29, 31, 94, 50, 142],
    'GoPro angle 2': [123, 157, 284, 126, 187, 239, 20, 70, 59, 112],
    'GoPro angle 3': [145, 178, 168, 302, 168, 119, 19, 57, 40, 164],
    'PP angle 1': [342, 377, 632, 231, 286, 1113, 45, 117, 100, 219],
    'PP angle 2': [370, 449, 392, 164, 204, 783, 28, 99, 57, 209],
    'PP angle 3': [342, 356, 590, 322, 230, 315, 34, 117, 63, 159],
    'Human': [693, 793, 1965, 329, 588, 2424, 64, 230, 229, 489]
}
final_df = pd.DataFrame(all_together)

""" 
# Prepare the data
final_df = final_df.reset_index().melt(id_vars=["index"], var_name="Device_Angle", value_name="Count")
final_df["Device"] = final_df["Device_Angle"].apply(lambda x: x.split()[0])
final_df["Angle"] = final_df["Device_Angle"].apply(lambda x: x.split()[2] if len(x.split()) > 2 else "Human")

print('findal_df')
print(final_df)

# ANOVA
model = ols('Count ~ C(Device) + C(Angle) + C(Device):C(Angle)', data=final_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Inspect the model summary
print("\nModel summary:")
print(model.summary())

# Check if we have significant results
print("ANOVA results:")
print(anova_table)

# If significant, proceed with Tukey's HSD
if anova_table["PR(>F)"][0] < 0.05 or anova_table["PR(>F)"][1] < 0.05:
    print("\nPerforming Tukey's HSD...\n")
    tukey = pairwise_tukeyhsd(endog=final_df['Count'], groups=final_df['Device_Angle'], alpha=0.05)
    print(tukey)

print('tukey.summary()')
print(tukey.summary())

#Save summary into a txt
with open('interval_plot_statistical_star_v3.txt', 'w') as f:
    f.write(anova_table.to_string())
    f.write(tukey.summary().as_text())

"""
#####################################

# DEVICE FACTOR

final_df = pd.DataFrame(all_together)

# Prepare the data
final_df = final_df.reset_index().melt(id_vars=["index"], var_name="Device_Angle", value_name="Count")
final_df["Device"] = final_df["Device_Angle"].apply(lambda x: x.split()[0])
final_df["Angle"] = final_df["Device_Angle"].apply(lambda x: x.split()[2] if len(x.split()) > 2 else "Human")

final_df.to_csv('Figure_2_test.csv', index=False)
# ANOVA
model = ols('Count ~ C(Device) + C(Angle) + C(Device):C(Angle)', data=final_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Check if we have significant results
print("ANOVA results:")
print(anova_table)

# If significant, proceed with Tukey's HSD
if anova_table["PR(>F)"][0] < 0.05 or anova_table["PR(>F)"][1] < 0.05:
    print("\nPerforming Tukey's HSD...\n")
    tukey = pairwise_tukeyhsd(endog=final_df['Count'], groups=final_df['Device_Angle'], alpha=0.05)
    print(tukey)
    with open('interval_plot_statistical_star_v3.txt', 'w') as f:
        f.write(tukey.summary().as_text())

# Additional ANOVA for 'Device' only
model_device = ols('Count ~ C(Device)', data=final_df).fit()
anova_table_device = sm.stats.anova_lm(model_device, typ=2)

print("\nANOVA results for Device:")
print(anova_table_device)

# If significant, proceed with Tukey's HSD for 'Device'
if anova_table_device["PR(>F)"][0] < 0.05:
    print("\nPerforming Tukey's HSD for Device...\n")
    tukey_device = pairwise_tukeyhsd(endog=final_df['Count'], groups=final_df['Device'], alpha=0.05)
    print(tukey_device)
    with open('DEVICE FACTOR.txt', 'w') as f:
        # save the anova results
        f.write(anova_table_device.to_string())
        f.write(tukey_device.summary().as_text())


# Check for missing values
print(final_df.isnull().sum())

# Check data types
print(final_df.dtypes)

# Inspect data distribution
print(final_df['Device'].value_counts())
print(final_df['Count'].describe())