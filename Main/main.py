import torch
import time
import numpy as np
from torch import nn
from torch.autograd import Variable
from sklearn import metrics
from torch.optim import lr_scheduler
from tensorboard_logger import configure, log_value
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support
import os
import pandas as pd
from model_build import SpectralCRNN_Reg_Dropout, SpectralCRNN
from sklearn.model_selection import train_test_split



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
#train_data_dir = os.path.join(project_dir, 'Data', 'train_data')

#ONLY_ISMIR_2012
train_data_dir = os.path.join(project_dir, 'onsets_ISMIR_2012', 'csv_large_label')

# Use a list comprehension to create a list of all CSV file paths
#csv_files_train = [f'{train_data_dir}/{file}' for file in os.listdir(train_data_dir) if file.endswith('.csv')]

# Use a list comprehension to create a list of all CSV file paths, limiting to the first two
csv_files = [f'{train_data_dir}/{file}' for file in os.listdir(train_data_dir) if file.endswith('.csv')]

print(len(csv_files)) #CHECK THE NUMBER OF CSV FILES AND ADJUST THE SLICING BELOW

csv_files_train_indices = [176, 175, 135, 27, 105, 109, 72, 123, 8, 71, 110, 55, 7, 64, 54, 90, 17, 80, 138, 181, 151, 148, 74, 18, 58, 171, 188, 166, 46, 196, 66, 117, 146, 106, 36, 40, 48, 163, 149, 184, 83, 82, 28, 200, 172, 2, 157, 122, 120, 79, 98, 189, 67, 154, 37, 52, 31, 127, 161, 15, 12, 60, 165, 102, 75, 155, 115, 1, 187, 132, 49, 158, 9, 6, 21, 50, 85, 152, 91, 92, 76, 29, 142, 53, 47, 3, 11, 167, 99, 111, 139, 133, 137, 199, 131, 164, 51, 194, 134, 42, 100, 197, 124, 177, 160, 56, 61, 114, 186, 153, 65, 23, 14, 183, 96, 125, 192, 173, 170, 32]
csv_files_train = [csv_files[i-1] for i in csv_files_train_indices]
csv_files_holdout_indices = [81, 119, 118, 5, 159, 150, 198, 84, 108, 103, 57, 104, 73, 107, 143, 190, 156, 128, 97, 94, 129, 113, 16, 185, 78, 147, 33, 168, 191, 121, 95, 162, 89, 144, 20, 38, 178, 35, 193, 24]
csv_files_holdout = [csv_files[i-1] for i in csv_files_holdout_indices]


# Use a dictionary comprehension to read each CSV file into a DataFrame
# The keys of the dictionary will be the file names, and the values will be the DataFrames
train_dataframes = {file: pd.read_csv(file) for file in csv_files_train}

# Use a dictionary comprehension to read each CSV file into a DataFrame
# The keys of the dictionary will be the file names, and the values will be the DataFrames
hold_out_dataframes = {file: pd.read_csv(file) for file in csv_files_holdout}

print(hold_out_dataframes.keys(), train_dataframes.keys())

#ONLY FOR ONSET_ISMIR - ADJUST THE SLICING csv_files[:]
csv_files_test_indices = [145, 70, 10, 44, 34, 116, 63, 136, 43, 112, 68, 62, 45, 195, 69, 86, 13, 140, 169, 130, 22, 39, 25, 88, 19, 93, 4, 30, 41, 101, 59, 141, 179, 26, 182, 126, 180, 174, 87, 77]
csv_files_test = [csv_files[i-1] for i in csv_files_test_indices]

# Use a dictionary comprehension to read each CSV file into a DataFrame
# The keys of the dictionary will be the file names, and the values will be the DataFrames
test_dataframes = {file: pd.read_csv(file) for file in csv_files_test}

'''for filename, df in train_split.items():
    train_df, hold_out_df = train_test_split(df, test_size=0.1, random_state=17)  # Adjust test_size as needed
    train_dataframes[filename] = train_df
    hold_out_dataframes[filename] = hold_out_df'''

#test_data_dir = os.path.join(project_dir, 'Data', 'test_data')

# Use a list comprehension to create a list of all CSV file paths
#csv_files_test = [f'{test_data_dir}/{file}' for file in os.listdir(test_data_dir) if file.endswith('.csv')]

# Use a list comprehension to create a list of all CSV file paths, limiting to the first two
#csv_files_test = [f'{test_data_dir}/{file}' for file in os.listdir(test_data_dir) if file.endswith('.csv')][:1]

#test_dataframes = {file: pd.read_csv(file) for file in csv_files_test}


# WARNING - TEST CODE ONLY
# List all files in the directory and get their sizes
'''csv_files_test = [(file, os.path.getsize(os.path.join(test_data_dir, file)))
             for file in os.listdir(test_data_dir)
             if file.endswith('.csv')]

# Sort files based on size (second element of the tuple)
csv_files_test.sort(key=lambda x: x[1])

# Select the smallest file (first in the sorted list)
if csv_files_test:  # Ensure there are files in the list
    smallest_file = csv_files_test[0][0]  # Correctly reference the sorted list
    smallest_file_path = os.path.join(test_data_dir, smallest_file)
    # Load the smallest file into a DataFrame and store it in a dictionary with the file name as the key
    test_dataframes = {smallest_file: pd.read_csv(smallest_file_path)}
    print(f"Loaded smallest file: {smallest_file}")
else:
    print("No CSV files found in the directory.")'''
#TEST CODE ENDS HERE

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

# Determine the maximum length of spectrograms across all dataframes
max_length = max(df['Spectrogram'].apply(lambda x: len(x.strip('[]').split())).max() for df in hold_out_dataframes.values())
print(f"Maximum length calculated: {max_length}")

# Check before using compute_normalization_constants
print(f"Processing normalization constants with max_length: {max_length}")
mean, std = compute_normalization_constants(hold_out_dataframes, max_length)

# Processing dataframes
process_dataframes(train_dataframes, max_length)
process_dataframes(test_dataframes, max_length)

# Define Model
model = SpectralCRNN_Reg_Dropout()
# Define optimizer and loss function
criterion = nn.MSELoss()
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)


batch_time = AverageMeter()
data_time = AverageMeter()


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

    for file, df in train_dataframes.items():

        # Time and loss updates - Our metrics
        start_time = time.time()

        # Inspect data before stacking to ensure there are no objects
        if any(isinstance(x, np.ndarray) and x.dtype == object for x in df['Spectrogram']):
            print("Object dtype found in Spectrogram arrays")

        inputs = np.vstack(df['Spectrogram'].values)
        # Double-check inputs dtype and convert to float32 if not already
        if inputs.dtype != np.float32:
            inputs = inputs.astype(np.float32)

        # Convert to PyTorch tensor
        try:
            inputs = torch.tensor(inputs).unsqueeze(1)  # Ensure correct shape [N, C, H, W]
        except TypeError as e:
            print("Failed to convert inputs to a tensor:", e)
            print("Inputs dtype:", inputs.dtype)
            break  # Break to avoid proceeding with incorrect data



        # Ensure dimensions [batch_size, channels, height, width]
        inputs = inputs.view(-1, 1, 254, 1)


        targets = torch.tensor(df['onset'].values, dtype=torch.float32)       # Assuming 'onset' contains the target values
        data_time.update(time.time() - end)
        inputs = Variable(inputs, requires_grad=False)
        targets = Variable(targets, requires_grad=False)
        targets = targets.view(-1, 1)

        # Initialize hidden state
        model.init_hidden(inputs.size(0))

        optimizer.zero_grad()

        # Forward pass
        out = model.forward(inputs)


        probabilities = torch.sigmoid(out).detach().cpu().numpy()
        all_predictions.extend(out.data.cpu().numpy())
        all_targets.extend(targets.data.cpu().numpy())
        all_predictions_prob.extend(probabilities)


        loss = criterion(out, targets)

        loss.backward()
        optimizer.step()

        # Time and loss updates - Our metrics
        batch_time.update(time.time() - start_time)
        losses.update(loss.item(), inputs.size(0))


        #Optional: print progress for the batch
        print('Epoch: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {losses.val:.4f} ({losses.avg:.4f})'.format(
            epoch + 1, num_epochs, batch_time=batch_time, losses=losses))

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

    for file, df in test_dataframes.items():  # Loop over test dataframes

        # Inspect data before stacking to ensure there are no objects
        if any(isinstance(x, np.ndarray) and x.dtype == object for x in df['Spectrogram']):
            print("Object dtype found in Spectrogram arrays")

        try:
            inputs = np.vstack(df['Spectrogram'].values)
            inputs = inputs.astype(np.float32)  # Ensure that the data type is float32

            # Convert to PyTorch tensor and adjust shape
            inputs = torch.tensor(inputs).unsqueeze(1)  # Add a channel dimension
            inputs = inputs.view(-1, 1, 254, 1)  # Reshape to match the expected input of the model

            targets = df['onset'].values
            targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

            model.init_hidden(inputs.size(0))  # Initialize hidden state for the model
            model.eval()  # Set the model to evaluation mode

            with torch.no_grad():  # Disables gradient calculation to save memory and computations
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                validation_losses.update(loss.item(), inputs.size(0))

                probabilities = torch.sigmoid(outputs).detach().cpu().numpy()
                all_val_predictions.extend(outputs.cpu().numpy())
                all_val_targets.extend(targets.cpu().numpy())
                all_val_probs.extend(probabilities)

        except TypeError as e:
            print(f"Failed to convert inputs to a tensor: {e}")
            print("Inputs dtype:", inputs.dtype)
            continue  # Skip this iteration and continue with the next file

        # Optional: print progress for each batch in the validation loop
        print('Validation - Epoch: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {validation_losses.val:.4f} ({validation_losses.avg:.4f})'.format(
            epoch + 1, num_epochs, batch_time=batch_time, validation_losses=validation_losses))

        total_val_loss += loss.item() * inputs.size(0)



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
                pred_probabilities=np.array(all_val_probs)
                # Ensure this is the probability of the positive class
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

