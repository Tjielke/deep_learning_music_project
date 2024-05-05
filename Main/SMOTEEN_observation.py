import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Get the project directory path
project_dir = os.getcwd()

# Join the project directory path with the train_data directory
train_data_dir = os.path.join(project_dir, 'Data', 'train_data')

# Use a list comprehension to create a list of all CSV file paths
#csv_files = [f'{train_data_dir}/{file}' for file in os.listdir(train_data_dir) if file.endswith('.csv')]

# Use a list comprehension to create a list of all CSV file paths, limiting to the first two
csv_files = [f'{train_data_dir}/{file}' for file in os.listdir(train_data_dir) if file.endswith('.csv')][:2]

# Use a dictionary comprehension to read each CSV file into a DataFrame
# The keys of the dictionary will be the file names, and the values will be the DataFrames
train_split = {file: pd.read_csv(file) for file in csv_files}

# Assuming train_dataframes is a dictionary where the key is the filename and the value is the DataFrame
train_dataframes = {}
hold_out_dataframes = {}

for filename, df in train_split.items():
    train_df, hold_out_df = train_test_split(df, test_size=0.5, random_state=17)  # Adjust test_size as needed
    train_dataframes[filename] = train_df
    hold_out_dataframes[filename] = hold_out_df

def custom_spectrogram_parser(spect_str, max_length):
    try:
        array = np.array([float(num) for num in spect_str.strip('[]').split() if num.strip()])
        if array.size < max_length:
            array = np.pad(array, (0, max_length - array.size), mode='constant')
        elif array.size > max_length:
            array = array[:max_length]
        return array
    except Exception as e:
        print(f"Error parsing spectrogram: {e}")
        return np.zeros(max_length)  # Ensure consistent shape

def normalize_spectrograms(spect, mean, std, epsilon=1e-10):
    normalized_spect = (spect - mean) / (std + epsilon)  # Adding epsilon to avoid division by zero
    return normalized_spect

def compute_normalization_constants(dataframes, max_length):
    all_spectrograms = []
    for df in dataframes.values():
        spectrograms = df['Spectrogram'].apply(lambda x: custom_spectrogram_parser(x, max_length))
        for spec in spectrograms:
            if len(spec) != max_length:
                print(f"Error: Array length {len(spec)} does not match max_length {max_length}")
            all_spectrograms.append(spec)
    all_spectrograms = np.vstack(all_spectrograms)
    mean = np.mean(all_spectrograms, axis=0)
    std = np.std(all_spectrograms, axis=0)
    return mean, std

def plot_class_distribution(labels, title):
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(8, 4))
    plt.bar(unique, counts, color=['red', 'green'])
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(unique)
    plt.show()

def process_dataframes(dataframes, max_length):
    for filename, df in dataframes.items():
        # Apply parsing and normalization transformations
        df['Spectrogram'] = df['Spectrogram'].apply(lambda x: custom_spectrogram_parser(x, max_length))
        df['Spectrogram'] = df['Spectrogram'].apply(lambda x: normalize_spectrograms(x, mean, std, epsilon=1e-10))
        dataframes[filename] = df

def stack_features_and_labels(dataframes):
    features = []
    labels = []
    for df in dataframes.values():
        if df['Spectrogram'].apply(len).unique()[0] != max_length:
            print(f"Error: Not all spectrogram lengths in {df} match max_length {max_length}")
        features.extend(df['Spectrogram'].tolist())
        labels.extend(df['onset'].tolist())
    return np.array(features), np.array(labels)


# Determine the maximum length of spectrograms across all dataframes
max_length = max(df['Spectrogram'].apply(lambda x: len(x.strip('[]').split())).max() for df in hold_out_dataframes.values())
print(f"Maximum length calculated: {max_length}")

# Check before using compute_normalization_constants
print(f"Processing normalization constants with max_length: {max_length}")
mean, std = compute_normalization_constants(hold_out_dataframes, max_length)

# Processing dataframes
process_dataframes(train_dataframes, max_length)
process_dataframes(hold_out_dataframes, max_length)


# Example function to flatten and then reshape data
def apply_custom_smoteenn(features, labels, smote_percentage):
    # Calculate the desired ratio based on the percentage
    majority_class_count = np.bincount(labels).max()
    minority_class_count = np.bincount(labels).min()
    if smote_percentage > 0:
        target_minority_count = int(majority_class_count * (1 + smote_percentage / 100))
    else:
        target_minority_count = minority_class_count

    # Calculate the number of samples to generate
    samples_to_generate = target_minority_count - minority_class_count

    if samples_to_generate <= 0:
        print("No SMOTE needed as the requested percentage does not lead to minority oversampling.")
        return features, labels

    sampling_strategy = {1: target_minority_count}  # Assuming class '1' is the minority class
    smote = SMOTEENN(sampling_strategy=sampling_strategy, random_state=42)
    features_resampled, labels_resampled = smote.fit_resample(features, labels)
    return features_resampled, labels_resampled

def apply_smoteenn(features, labels):
    # Print original class distribution
    print(f"Original class distribution: {np.unique(labels, return_counts=True)}")
    smote_enn = SMOTEENN(random_state=42)
    features_resampled, labels_resampled = smote_enn.fit_resample(features, labels)
    # Print resampled class distribution
    print(f"Resampled class distribution: {np.unique(labels_resampled, return_counts=True)}")
    return features_resampled, labels_resampled


def apply_smote(features, labels, percentage=100):
    # Print original class distribution
    original_counts = np.bincount(labels)
    print(f"Original class distribution: {dict(enumerate(original_counts))}")

    # Adjust the sampling strategy based on the percentage
    if percentage > 0:
        majority_count = max(original_counts)
        minority_class = np.argmin(original_counts)
        # Calculate the number of minority samples to generate
        num_minority_samples = int(
            original_counts[minority_class] + (majority_count - original_counts[minority_class]) * (percentage / 100))
        sampling_strategy = {minority_class: num_minority_samples}
    else:
        sampling_strategy = 'minority'

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    features_resampled, labels_resampled = smote.fit_resample(features, labels)

    # Print resampled class distribution
    resampled_counts = np.bincount(labels_resampled)
    print(f"Resampled class distribution: {dict(enumerate(resampled_counts))}")

    return features_resampled, labels_resampled

# Stack features and labels from the processed train dataframes
train_features, train_labels = stack_features_and_labels(train_dataframes)
print("Shapes:", train_features.shape, train_labels.shape)


def process_and_analyze_data(dataframes, max_length):
    all_features = []
    all_labels = []

    for filename, df in dataframes.items():
        features = np.stack(df['Spectrogram'].values)
        labels = df['onset'].values

        all_features.append(features)
        all_labels.append(labels)

    all_features = np.concatenate(all_features)
    all_labels = np.concatenate(all_labels)

    # Plot initial class distribution
    plot_class_distribution(all_labels, "Initial Class Distribution")

    # Apply SMOTEENN
    #resampled_features, resampled_labels = apply_custom_smoteenn(all_features.reshape(all_features.shape[0], -1), all_labels,smote_percentage=100)

    #resampled_features, resampled_labels = apply_smoteenn(all_features.reshape(all_features.shape[0], -1), all_labels)


    resampled_features, resampled_labels = apply_smote(all_features.reshape(all_features.shape[0], -1), all_labels,percentage=100)

    # Plot class distribution after SMOTEENN
    #plot_class_distribution(resampled_labels, "Class Distribution After SMOTEENN")

    # Plot class distribution after SMOTE
    plot_class_distribution(resampled_labels, "Class Distribution After SMOTE")


    return resampled_features, resampled_labels



process_and_analyze_data(train_dataframes, max_length)
