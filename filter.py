import pandas as pd

# Load the original CSV
input_file = "derm12345_labels_with_nev.csv"  # replace with your actual filename
output_file = "derm12345_labels.csv"  # name for the new filtered CSV

# Read the CSV into a DataFrame
df = pd.read_csv(input_file)

# Remove rows where NEV == 1
filtered_df = df[df["NEV"] != 1]

# Drop the NEV column
filtered_df = filtered_df.drop(columns=["NEV"])

# Save the filtered DataFrame to a new CSV (original stays unchanged)
filtered_df.to_csv(output_file, index=False)

print(f"Filtered file saved as: {output_file}")
