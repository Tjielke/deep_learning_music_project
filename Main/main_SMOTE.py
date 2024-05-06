import torch
import time
import numpy as np
from torch import nn
from torch.autograd import Variable
from sklearn import metrics
from torch.optim import lr_scheduler
import torch.nn.functional as F
from tensorboard_logger import configure, log_value
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support
import os
import pandas as pd
from model_build import SpectralCRNN_Reg_Dropout, SpectralCRNN
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import datetime


# Get the current time
current_time = datetime.datetime.now()

# Print the current time
print("Program started at:", current_time.strftime("%Y-%m-%d %H:%M:%S"))



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def evaluate_classification(targets, predictions, pred_probabilities=None):
    r2 = metrics.r2_score(targets, predictions)
    targets = np.round(targets).astype(int)
    predictions = 1 / (1 + np.exp(-predictions))  # Apply sigmoid function
    predictions = np.round(predictions).astype(int)
    accuracy = metrics.accuracy_score(targets, predictions) * 100

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(targets, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted', zero_division=0
    )

    # Calculate ROC AUC if probability scores are provided
    roc_auc = None
    if pred_probabilities is not None:
        try:
            # Assuming binary classification; adjust for multiclass if necessary
            roc_auc = roc_auc_score(targets, pred_probabilities)
        except ValueError as e:
            print(f"ROC AUC calculation error: {e}")

    return r2, accuracy, conf_matrix, precision, recall, f1, roc_auc


# Get the project directory path
project_dir = os.getcwd()

# Join the project directory path with the train_data directory
train_data_dir = os.path.join(project_dir, 'Data', 'train_data')

# Use a list comprehension to create a list of all CSV file paths
#csv_files_train = [f'{train_data_dir}/{file}' for file in os.listdir(train_data_dir) if file.endswith('.csv')]

# Use a list comprehension to create a list of all CSV file paths, limiting to the first two
csv_files = [f'{train_data_dir}/{file}' for file in os.listdir(train_data_dir) if file.endswith('.csv')]

csv_files_train = csv_files[:1]
csv_files_holdout = csv_files[9:10]


# Use a dictionary comprehension to read each CSV file into a DataFrame
# The keys of the dictionary will be the file names, and the values will be the DataFrames
train_dataframes = {file: pd.read_csv(file) for file in csv_files_train}

# Use a dictionary comprehension to read each CSV file into a DataFrame
# The keys of the dictionary will be the file names, and the values will be the DataFrames
hold_out_dataframes = {file: pd.read_csv(file) for file in csv_files_holdout}

print(hold_out_dataframes.keys(), train_dataframes.keys())


'''for filename, df in train_split.items():
    train_df, hold_out_df = train_test_split(df, test_size=0.1, random_state=17)  # Adjust test_size as needed
    train_dataframes[filename] = train_df
    hold_out_dataframes[filename] = hold_out_df'''

test_data_dir = os.path.join(project_dir, 'Data', 'test_data')

# Use a list comprehension to create a list of all CSV file paths
#csv_files_test = [f'{test_data_dir}/{file}' for file in os.listdir(test_data_dir) if file.endswith('.csv')]

# Use a list comprehension to create a list of all CSV file paths, limiting to the first two
csv_files_test = [f'{test_data_dir}/{file}' for file in os.listdir(test_data_dir) if file.endswith('.csv')][:1]

test_dataframes = {file: pd.read_csv(file) for file in csv_files_test}

# Configure tensorboard logger
configure('runs/MelSpec_reg_lr0.0001_big_ELU_Adam_noteacc', flush_secs=2)

# Function to convert JSON-encoded strings back to numpy arrays
# Very important function that was  customized to fit our data. It won't work otherwise
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
process_dataframes(test_dataframes, max_length)


# Define batch size
batch_size = 64

# Example function to flatten and then reshape data
def apply_custom_smote(features, labels, smote_percentage):
    # Calculate the desired ratio based on the percentage
    original_counts = np.bincount(labels)
    print(f"Original class distribution: {dict(enumerate(original_counts))}")
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
    smote = SMOTEENN(sampling_strategy=sampling_strategy, random_state=17)
    features_resampled, labels_resampled = smote.fit_resample(features, labels)

    # Print resampled class distribution
    resampled_counts = np.bincount(labels_resampled)
    print(f"Resampled class distribution: {dict(enumerate(resampled_counts))}")

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

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=17)
    features_resampled, labels_resampled = smote.fit_resample(features, labels)

    # Print resampled class distribution
    resampled_counts = np.bincount(labels_resampled)
    print(f"Resampled class distribution: {dict(enumerate(resampled_counts))}")

    return features_resampled, labels_resampled

# Stack features and labels from the processed train dataframes
train_features, train_labels = stack_features_and_labels(train_dataframes)
print("Shapes:", train_features.shape, train_labels.shape)

# Flatten features for SMOTE
train_features_flat = train_features.reshape(train_features.shape[0], -1)
# Apply SMOTE at 100%
X_resampled, y_resampled = apply_custom_smote(train_features_flat, train_labels, 100)
# Reshape for model input
X_resampled = X_resampled.reshape(-1, 1, max_length, X_resampled.shape[1] // max_length)

# Define Model
model = SpectralCRNN_Reg_Dropout()
# Define optimizer and loss function
criterion = nn.MSELoss()
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)


batch_time = AverageMeter()
data_time = AverageMeter()


train_loss = 0
validation_loss = 0

# Epoch is set into 1 because it is very slow at the moment. It is just for testing purposes

num_epochs = 20
best_val = 0.0

print("Training model...")
for epoch in range(num_epochs):
    # Training loop

    end = time.time()
    all_predictions = []
    all_targets = []
    all_predictions_prob = []

    losses = AverageMeter()

    model.train()  # Set model to training mode
    batch_time.reset()
    data_time.reset()
    total_loss = 0
    total_samples = 0

    # Randomize data for each epoch
    permutation = np.random.permutation(X_resampled.shape[0])

    for i in range(0, X_resampled.shape[0], batch_size):
        start_time = time.time()

        indices = permutation[i:i + batch_size]

        batch_features = X_resampled[indices]
        batch_labels = y_resampled[indices]

        # Convert numpy arrays to torch tensors
        inputs = torch.tensor(batch_features,
                              dtype=torch.float32)  # Should have shape [batch_size, 1, 254, input_width]
        targets = torch.tensor(batch_labels, dtype=torch.float32).view(-1,
                                                                       1)  # Ensure targets are the correct shape for loss computation

        # Initialize hidden state
        model.init_hidden(inputs.size(0))

        # Zero gradients before running the forward pass
        optimizer.zero_grad()

        # Forward pass
        out = model.forward(inputs)  # Ensure inputs are used here instead of targets

        probabilities = torch.sigmoid(out).detach().cpu().numpy()
        all_predictions.extend(out.data.cpu().numpy())
        all_targets.extend(targets.data.cpu().numpy())
        all_predictions_prob.extend(probabilities)

        # Calculate loss
        loss = criterion(out, targets)

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), inputs.size(0))
        batch_time.update(time.time() - start_time)
        start_time = time.time()


        #Optional: print progress for the batch
        ''' print('Epoch: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {losses.val:.4f} ({losses.avg:.4f})'.format(
            epoch + 1, num_epochs, batch_time=batch_time, losses=losses))'''

    pred_prob = np.array(all_predictions_prob)
    if len(all_predictions) > 0 and len(all_targets) > 0:
        # Check if probabilities need reshaping
        if pred_prob.ndim == 2 and pred_prob.shape[1] == 1:
            probs = pred_prob.squeeze(1)  # Removes the second dimension if it's unnecessary
        else:
            probs = pred_prob

        train_r2, train_accuracy, train_conf_matrix, train_precision, train_recall, train_f1, train_roc_auc = evaluate_classification(
            targets=np.array(all_targets),
            predictions=np.array(all_predictions),
            pred_probabilities=probs
        )
        print(f"Epoch {epoch + 1}/{num_epochs}, Training R2: {train_r2:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"Confusion Matrix:\n{train_conf_matrix}")
        print(f"Training Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-Score: {train_f1:.4f}")
        if train_roc_auc is not None:
            print(f"Training ROC AUC: {train_roc_auc:.4f}")
    else:
        print("No predictions or targets available for evaluation.")

    # Log metrics (if using TensorBoard or any other logging tool)
    log_value('Train Loss', losses.avg, epoch)
    log_value('Training R2', train_r2, epoch)
    log_value('Training Accuracy', train_accuracy, epoch)
    log_value('Training Precision', train_precision, epoch)
    log_value('Training Recall', train_recall, epoch)
    log_value('Training F1-Score', train_f1, epoch)


    scheduler.step()

    # Validation loop
    model.eval()

    # Github metrics - not used
    losses = AverageMeter()

    validation_losses = AverageMeter()  # Reset validation loss meter
    all_val_predictions = []
    all_val_targets = []
    all_val_probs = []

    #our metrics - not used
    total_val_loss = 0

    for file, df in test_dataframes.items():
        try:
            inputs = np.vstack(df['Spectrogram'].values)
            inputs = inputs.astype(np.float32)  # Ensures that the input is float32
            inputs = torch.tensor(inputs).unsqueeze(1)  # Convert to tensor and add channel dimension
            inputs = inputs.view(-1, 1, 254, 1)  # Ensure correct shape

            targets = df['onset'].values
            targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

            model.init_hidden(inputs.size(0))  # Initialize hidden state for the model
            model.eval()  # Set the model to evaluation mode

            with torch.no_grad():  # Disable gradient computation for validation
                outputs = model(inputs)
                probabilities = torch.sigmoid(outputs).detach().cpu().numpy()
                loss = criterion(outputs, targets)
                validation_losses.update(loss.item(), inputs.size(0))

                all_val_predictions.extend(outputs.cpu().numpy())
                all_val_targets.extend(targets.cpu().numpy())
                all_val_probs.extend(probabilities)

        except Exception as e:
            print(f"An error occurred while processing file {file}: {e}")

        '''# Optional: print progress for each batch in the validation loop
        print('Validation - Epoch: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {validation_losses.val:.4f} ({validation_losses.avg:.4f})'.format(
            epoch + 1, num_epochs, batch_time=batch_time, validation_losses=validation_losses))'''

        total_val_loss += loss.item() * inputs.size(0)

    # Inside your validation loop:

    # Convert list of probabilities to an appropriate format if needed
    probs_val = np.array(all_val_probs)
    if len(all_val_predictions) > 0 and len(all_val_targets) > 0:
        # Check if probabilities need reshaping
        if probs_val.ndim == 2 and probs_val.shape[1] == 1:
            probs = probs_val.squeeze(1)  # Removes the second dimension if it's unnecessary
        else:
            probs = probs_val

        # Ensure all_predictions_prob is the array of prediction probabilities
        val_r2, val_accuracy, val_conf_matrix, val_precision, val_recall, val_f1, val_roc_auc = evaluate_classification(
            targets=np.array(all_val_targets),
            predictions=np.array(all_val_predictions),
            pred_probabilities=np.array(all_val_probs)  # Ensure this is the probability of the positive class
        )
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation R2: {val_r2:.4f}, Accuracy: {val_accuracy:.2f}%")
        print(f"Confusion Matrix:\n{val_conf_matrix}")
        print(f"Validation Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}")
        if val_roc_auc is not None:
            print(f"Validation ROC AUC: {val_roc_auc:.4f}")
    else:
        print("No predictions or targets available for evaluation.")

    # Log validation metrics
    log_value('Validation Loss', validation_losses.avg, epoch)
    log_value('Validation R2', val_r2, epoch)
    log_value('Validation Accuracy', val_accuracy, epoch)
    log_value('Validation Precision', val_precision, epoch)
    log_value('Validation Recall', val_recall, epoch)
    log_value('Validation F1-Score', val_f1, epoch)

    if val_r2 > best_val:
        best_val = val_r2
        torch.save(model.state_dict(), 'model_SpectralCRNN_reg_lr0.0001_big_ELU_Adam_noteacc.pth')

# Get the current time
current_time = datetime.datetime.now()

# Print the current time
print("Program finished at:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
