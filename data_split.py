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
#-----------------------------------
#calculate the recurrence years 
#------------------------------------

from datetime import datetime

# Calculate recurrence years
df['recurrence_years'] = df.apply(
    lambda row: (pd.to_datetime(row['Date_of_recurrence']) - pd.to_datetime(row['Date_of_Surgery'])).days / 365.25 
    if row['Recurrence'] == 'yes' 
    else (pd.to_datetime(row['Last_news_date']) - pd.to_datetime(row['Date_of_Surgery'])).days / 365.25,
    axis=1
)

#count the total number of rows where the number of years it not na 
rows_before = len(df) - df['recurrence_years'].isna().sum()
print(rows_before)

# Print first 10 lines of the updated dataframe
print(df.head(10))

# Check for zero or negative years
invalid_years = df[df['recurrence_years'] <= 0]
if not invalid_years.empty:
    print("Warning: Found rows with zero or negative years:")
    print(invalid_years)
    # Optional: You might want to handle these cases
    # For example, you could set them to NaN
    df.loc[df['recurrence_years'] <= 0, 'recurrence_years'] = np.nan


#count the total number of rows where the number of years it not na after removing 0 or negative values
rows_after = len(df) - df['recurrence_years'].isna().sum()
print(rows_after)
print(df["recurrence_years"])

print("Lowest value:", df['recurrence_years'].min())
print("Highest value:", df['recurrence_years'].max())


# %%

# Determine number of bins using Sturges' rule
n_bins = int(np.ceil(np.log2(len(df)) + 1))

# Create bins using equal-width binning
df['disc_label'] = pd.cut(df['recurrence_years'], 
                           bins=n_bins, 
                           labels=False)  # This ensures 0 to n-1 labeling


print(df.head(10))
df.to_csv("intermediate.csv", index="False")

# %%

#---------------------------------
# add the censorship column  
# 0 -> event , 1-> no event 
#---------------------------------

df['censorship'] = (df['Recurrence'] == 'no').astype(int)

print(df[['censorship', 'Recurrence' ]])



#%%
#------------------------------
#prepare the train - test split
#------------------------------
print(df['Center'].value_counts())

df['split'] = df['Center'].apply(lambda x: 'training' if x in ['Bordeaux', 'Spain_Valencia'] else 'validation' if x == 'Muenster2' else 'test')


# %%
#------------------------------
#keep only the column needed
#rename the column "FILENAME"
#save the dataframe in a csv file
#------------------------------

# Select and rename columns
df_processed = df[['FILENAME', 'stage', 'recurrence_years', 'disc_label', 'censorship', 'PATIENT', 'split']].copy()
df_processed.rename(columns={'FILENAME': 'slide_id'}, inplace=True)

# Save to CSV
df_processed.to_csv('preprocess_table.csv', index=False)
# %%
