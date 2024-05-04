import os
import pandas as pd

# Get the project directory path
project_dir = os.getcwd()

# Function to load data and count values in 'onset' column
def load_data_and_count_values(data_dir):
    csv_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.csv')]
    dataframes = {file: pd.read_csv(file) for file in csv_files}
    onset_counts = {os.path.basename(file): df['onset'].value_counts() for file, df in dataframes.items() if 'onset' in df.columns}
    return onset_counts

# Training data analysis
train_data_dir = os.path.join(project_dir, 'Data', 'train_data')
train_onset_counts = load_data_and_count_values(train_data_dir)

# Test data analysis
test_data_dir = os.path.join(project_dir, 'Data', 'test_data')
test_onset_counts = load_data_and_count_values(test_data_dir)

# Create DataFrame for detailed counts
details = []

# Collecting training data details
for file, counts in train_onset_counts.items():
    details.append({
        "Dataset": "Training",
        "File": file,
        "Not Onset (0)": counts.get(0, 0),
        "Onset (1)": counts.get(1, 0)
    })

# Collecting test data details
for file, counts in test_onset_counts.items():
    details.append({
        "Dataset": "Testing",
        "File": file,
        "Not Onset (0)": counts.get(0, 0),
        "Onset (1)": counts.get(1, 0)
    })

details_df = pd.DataFrame(details)

# Aggregating total counts
total_counts = pd.DataFrame({
    "Dataset": ["Training", "Testing"],
    "Total Not Onset (0)": [
        sum(details_df[details_df["Dataset"] == "Training"]["Not Onset (0)"]),
        sum(details_df[details_df["Dataset"] == "Testing"]["Not Onset (0)"])
    ],
    "Total Onset (1)": [
        sum(details_df[details_df["Dataset"] == "Training"]["Onset (1)"]),
        sum(details_df[details_df["Dataset"] == "Testing"]["Onset (1)"])
    ]
})

# Combine detailed and total counts into a single Excel file with multiple sheets
with pd.ExcelWriter(os.path.join(project_dir, 'onset_counts_detailed_summary.xlsx')) as writer:
    details_df.to_excel(writer, sheet_name='Detailed Counts', index=False)
    total_counts.to_excel(writer, sheet_name='Total Counts', index=False)

print("Excel file with detailed and total counts has been saved.")
