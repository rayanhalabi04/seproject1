import os
import pandas as pd
# Paths
raw_data_path = "data/raw/ds_salaries.csv"
processed_data_path = "data/processed/cleaned_salaries.csv"
# Load data
df = pd.read_csv(raw_data_path)
"""
checking the data part 
df = pd.read_csv(raw_data_path)

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- SHAPE ---")
print(df.shape)

print("\n--- COLUMNS ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum()) no missing values were found

print("\n--- DUPLICATES ---")
print(df.duplicated().sum()) -> no duplicates were found
"""

print("\n--- ORIGINAL SHAPE ---")
print(df.shape)

#Drop useless columns
columns_to_drop = ["Unnamed: 0", "salary", "salary_currency"]
df = df.drop(columns=columns_to_drop, errors="ignore")

print("\n--- CLEANED SHAPE ---")
print(df.shape)

print("\n--- FINAL COLUMNS ---")
print(df.columns.tolist())

# Save cleaned dataset
os.makedirs("data/processed", exist_ok=True)
df.to_csv(processed_data_path, index=False)

print(f"\n Cleaned dataset saved at: {processed_data_path}")