import torch
import time
import numpy as np
from torch import nn
from torch.autograd import Variable
from sklearn import metrics
from torch.optim import lr_scheduler
from tensorboard_logger import configure, log_value
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
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

def evaluate_classification(targets, predictions):
    r2 = metrics.r2_score(targets, predictions)
    targets = np.round(targets*10).astype(int)
    predictions = predictions * 10
    predictions[predictions < 0] = 0
    predictions[predictions > 10] = 10
    predictions = np.round(predictions).astype(int)
    accuracy = metrics.accuracy_score(targets, predictions) * 100

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(targets, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted', zero_division=0
    )

    return r2, accuracy, conf_matrix, precision, recall, f1


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

test_data_dir = os.path.join(project_dir, 'Data', 'test_data')

# Use a list comprehension to create a list of all CSV file paths
#csv_files = [f'{test_data_dir}/{file}' for file in os.listdir(test_data_dir) if file.endswith('.csv')]

# Use a list comprehension to create a list of all CSV file paths, limiting to the first two
csv_files = [f'{test_data_dir}/{file}' for file in os.listdir(test_data_dir) if file.endswith('.csv')][:1]

test_dataframes = {file: pd.read_csv(file) for file in csv_files}

# Configure tensorboard logger
configure('runs/MelSpec_reg_lr0.0001_big_ELU_Adam_noteacc', flush_secs=2)

# Function to convert JSON-encoded strings back to numpy arrays
# Very important function that was  customized to fit our data. It won't work otherwise
def custom_spectrogram_parser(spect_str, max_length):
    if isinstance(spect_str, np.ndarray):
        return spect_str  # If already an array, return as is
    try:
        cleaned_str = spect_str.strip('[]')
        array = np.array([float(num) for num in cleaned_str.split() if num.strip()])
        # Ensure the array is padded to max_length
        if array.size < max_length:
            array = np.pad(array, (0, max_length - array.size), mode='constant')
        assert array.size == max_length, f"Array length after padding is {array.size}, expected {max_length}"
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
process_dataframes(hold_out_dataframes, max_length)
process_dataframes(test_dataframes, max_length)

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


for epoch in range(num_epochs):

    # Training loop

    end = time.time()
    all_predictions = []
    all_targets = []

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

        all_predictions.extend(out.data.cpu().numpy())
        all_targets.extend(targets.data.cpu().numpy())
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


    # Inside your training loop
    train_r2, train_accuracy, train_conf_matrix, train_precision, train_recall, train_f1 = evaluate_classification(
        np.array(all_targets), np.array(all_predictions))

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {losses.avg:.4f}, Training R2: {train_r2:.4f}, Training Accuracy: {train_accuracy:.2f}%")
    print(f"Training Confusion Matrix:\n{train_conf_matrix}")
    print(f"Training Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-Score: {train_f1:.4f}")

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

    #our metrics - not used
    total_val_loss = 0

    for file, df in test_dataframes.items():  # Loop over test dataframes

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
            break  # Break to avoid proceeding with incorrect data\

        inputs = inputs.view(-1, 1, 254, 1)

        targets = torch.tensor(df['onset'].values, dtype=torch.float32)  # Assuming 'onset' contains the target values
        data_time.update(time.time() - end)
        inputs = Variable(inputs, requires_grad=False)
        targets = Variable(targets, requires_grad=False)
        targets = targets.view(-1, 1)

        # Initialize hidden state
        model.init_hidden(inputs.size(0))

        # Forward pass
        out = model.forward(inputs)
        # Ensure dimensions [batch_size, channels, height, width]

        # Our metrics
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            validation_losses.update(loss.item(), inputs.size(0))
            all_val_predictions.extend(outputs.cpu().numpy())
            all_val_targets.extend(targets.cpu().numpy())

        # Optional: print progress for each batch in the validation loop
        print('Validation - Epoch: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {validation_losses.val:.4f} ({validation_losses.avg:.4f})'.format(
            epoch + 1, num_epochs, batch_time=batch_time, validation_losses=validation_losses))

        total_val_loss += loss.item() * inputs.size(0)


    # Inside your validation loop
    val_r2, val_accuracy, val_conf_matrix, val_precision, val_recall, val_f1 = evaluate_classification(
        np.array(all_val_targets), np.array(all_val_predictions))

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {validation_losses.avg:.4f}, Validation R2: {val_r2:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Validation Confusion Matrix:\n{val_conf_matrix}")
    print(f"Validation Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}")

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

