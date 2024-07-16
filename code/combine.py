# Import necessary packages
import pandas as pd
import os
import re

# Navigate to the folder where your CSV files are 
os.chdir(r"<file_location>")

# Create an empty dataframe
df = pd.DataFrame([])

# Read all CSV files and append them to df
csv_files = []
for root, dirs, files in os.walk("."):
    for name in files:
        if name.endswith(".csv"):
            csv_files.append(name)

# Sort the CSV file names based on the "ep" number
csv_files.sort(key=lambda x: int(re.search(r'ep(\d+)', x).group(1)) if re.search(r'ep(\d+)', x) else 0)
print(csv_files)

for file_name in csv_files:
    file_path = os.path.join(root, file_name)
    df_temp = pd.read_csv(file_path, on_bad_lines='skip')
    df = pd.concat([df, df_temp])

# Save df to a CSV file
df.to_csv('Combined.csv')
