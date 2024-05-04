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

# Print the results
sum_train_0 = 0
sum_train_1 = 0
sum_test_0 = 0
sum_test_1 = 0
print("Training Data Onset Counts:")
for file, counts in train_onset_counts.items():
    print(f"{file}:")
    print(counts)
    sum_train_0 += counts[0]
    sum_train_1 += counts[1]
    print()

print("Test Data Onset Counts:")
for file, counts in test_onset_counts.items():
    print(f"{file}:")
    print(counts)
    sum_test_0 += counts[0]
    sum_test_1 += counts[1]
    print()

print(f"Training Data Total Not Onset Counts:", {sum_train_0})
print(f"Training Data Total Onset Counts:", {sum_train_1})
print(f"Test Data Total Not Onset Counts:", {sum_test_0})
print(f"Test Data Total Onset Counts:", {sum_test_1})
