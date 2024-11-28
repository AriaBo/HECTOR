###############################
#FILE EXPLANATION - README
##############################
#deply_Italy_slide_table.csv --> smapleID and filename ITALY + SPAIN
#slide_table_2.csv --> slide table sampleID and filename ALL THE OTHER CENTRES

#merged_dataset --> clinical data ALL SAMPLES
#recurrence_pred --> clinical data recurrence only
#unsupervised_training.csv --> clinical data NO recurrence only

#recurrence_pred_with_filename --> same as recurrence_pred but with FILENAME 



#sample_file_match.csv --> sanmpleID and filename ALL 
#sample_file_match_filtered.csv --> sampleID and filename recurrence ONLY




###############################
#SELECT THE RIGHT SAMPLES FOR
#UNSUPERVISED LEARNING 
#HECTOR TRAINING
###############################

#%%
import pandas as pd

# Read the merged dataset
df = pd.read_csv('merged_dataset.csv')

# Filter for samples with 'no' in Adjuvant_TKI and 'yes' or 'no' in Recurrence
recurrence_pred = df[(df['Adjuvant_TKI'] == 'no') & (df['Recurrence'].isin(['yes', 'no']))]

# Save the filtered dataframe
recurrence_pred.to_csv('recurrence_pred.csv', index=False)

# Create the unsupervised training dataset (remaining rows)
unsupervised_training = df[~df.index.isin(recurrence_pred.index)]

# Save the unsupervised training dataset
unsupervised_training.to_csv('unsupervised_training.csv', index=False)

print(f"Recurrence prediction dataset shape: {recurrence_pred.shape}")
print(f"Unsupervised training dataset shape: {unsupervised_training.shape}")


# %%
##############################
#READ THE TWO FILES: slide_table_2.csv (all the centers no SPAIN and ITALY)
#and deploy_Italy_slide_table.csv (with ITALY + SPAIN)
##############################

import pandas as pd

# Read the two slide tables
slide_table_2 = pd.read_csv('slide_table_2.csv')
deploy_italy_slide_table = pd.read_csv('deploy_Italy_slide_table.csv')

# Merge the two tables, keeping all rows
sample_file_match = pd.concat([slide_table_2, deploy_italy_slide_table], ignore_index=True)

# Save the merged file
sample_file_match.to_csv('sample_file_match.csv', index=False)

# Read the recurrence prediction file
recurrence_pred = pd.read_csv('recurrence_pred.csv')

# Filter sample_file_match to keep only rows matching PATIENT in recurrence_pred
filtered_sample_file_match = sample_file_match[sample_file_match['PATIENT'].isin(recurrence_pred['PATIENT'])]

# Save the filtered file
filtered_sample_file_match.to_csv('sample_file_match_filtered.csv', index=False)

print(f"Original sample_file_match shape: {sample_file_match.shape}")
print(f"Filtered sample_file_match shape: {filtered_sample_file_match.shape}")

# %%
#######################################################################
#-----------------------------------------------------------------------
########################################################################

##############################
#PREPARE THE TABLE ACCRODING TO HECTOR README
##############################


#%%
#---------------------------
#1. merge the clinical table with the filename
#--------------------------

import pandas as pd

# Read the files
sample_file_match_filtered = pd.read_csv('/mnt/bulk-curie/arianna/HECTOR/sample_file_match_filtered.csv')
recurrence_pred = pd.read_csv('/mnt/bulk-curie/arianna/HECTOR/recurrence_pred.csv')

# Merge the dataframes based on the PATIENT column
merged_df = recurrence_pred.merge(sample_file_match_filtered[['PATIENT', 'FILENAME']], 
                                   on='PATIENT', 
                                   how='left')

# Save the merged dataframe
merged_df.to_csv('/mnt/bulk-curie/arianna/HECTOR/recurrence_pred_with_filename.csv', index=False)

print(f"Original recurrence_pred shape: {recurrence_pred.shape}")
print(f"Merged dataframe shape: {merged_df.shape}")



# %%

#-------------------------------
#2. add the stage column according to the mitotic index
#<= 5 LOW >5 HIGH
#-------------------------------
import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('/mnt/bulk-curie/arianna/HECTOR/recurrence_pred_with_filename.csv')

# Convert Mitotic_index_5mm2 to numeric, coercing errors to NaN
df['Mitotic_index_5mm2'] = pd.to_numeric(df['Mitotic_index_5mm2'], errors='coerce')

# Create the 'stage' column 
df['stage'] = np.where(
    df['Mitotic_index_5mm2'].isna(), 
    np.nan, 
    np.where(df['Mitotic_index_5mm2'] <= 5, 'low', 'high')
)

# Optional: Print the first few rows to verify
print(df[['Mitotic_index_5mm2', 'stage']].head(15))

# %%
