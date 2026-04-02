import pandas as pd
import glob
import os
import numpy as np

# 1. Setup paths
folder_path = 'archive (1)' 
all_files = glob.glob(os.path.join(folder_path, "*Values.csv"))
print(f"Found {len(all_files)} monthly files. Starting merge...")

all_data = []
for file in all_files:
    df = pd.read_csv(file, skiprows=1)
    all_data.append(df)

master_df = pd.concat(all_data, axis=0, ignore_index=True)

# 2. Robust Date Conversion
def convert_excel_date(date_val):
    try:
        # If it's a number like "43549"
        if str(date_val).isdigit() or isinstance(date_val, (int, float)):
            return pd.to_datetime(float(date_val), unit='D', origin='1899-12-30')
        # Otherwise treat as string
        return pd.to_datetime(date_val, dayfirst=True, errors='coerce')
    except:
        return pd.NaT

print("Converting dates and cleaning data...")
master_df['Collected'] = master_df['Collected'].apply(convert_excel_date)

# Drop rows where date couldn't be parsed
master_df = master_df.dropna(subset=['Collected'])

# 3. Sort and Clean
master_df = master_df.sort_values(by=['MedID', 'Collected'])
master_df.columns = [c.split(' (')[0].replace(' ', '_') for c in master_df.columns]

# 4. Split into 5 Hospitals
unique_patients = master_df['MedID'].unique()
np.random.shuffle(unique_patients)
hospital_groups = np.array_split(unique_patients, 5)
hospitals = ['Hospital_A', 'Hospital_B', 'Hospital_C', 'Hospital_D', 'Hospital_E']

for i, group in enumerate(hospital_groups):
    hospital_data = master_df[master_df['MedID'].isin(group)]
    file_name = f"{hospitals[i]}_Master_Records.csv"
    hospital_data.to_csv(file_name, index=False)
    print(f"Created {file_name} with {len(hospital_data)} records.")

print("\nSuccess! Error resolved and data split.")