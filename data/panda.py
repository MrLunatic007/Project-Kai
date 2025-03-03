import pandas as pd

# File paths
pre_data_file = "data/text.csv"  # Adjust if path differs
dataset_file = "data/dataset_3.csv"

# Load pre_data.csv
pre_data_df = pd.read_csv(pre_data_file)

# Display the DataFrame
print("Here’s the raw data from pre_data.csv:")
print(pre_data_df.head())  # Show first 5 rows
print("\nColumns available:", list(pre_data_df.columns))
print("\nShape (rows, columns):", pre_data_df.shape)

# Optional: Show a sample if it’s large
print("\nSample of 5 random rows:")
print(pre_data_df.sample(5))

# Columns to delete
columns_to_delete = ['0']

# Drop the specified columns from the DataFrame
pre_data_df = pre_data_df.drop(columns=columns_to_delete)

# Display the DataFrame after deletion
print("\nDataFrame after removing specified columns:")
print(pre_data_df.head())  # Show first 5 rows after removing columns

# Optional: Save the updated DataFrame to a new CSV file
pre_data_df.to_csv(dataset_file, index=False)

print(f"\nUpdated data has been saved to {dataset_file}")
