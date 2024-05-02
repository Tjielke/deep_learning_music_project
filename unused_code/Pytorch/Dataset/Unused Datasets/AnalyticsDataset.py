import os
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader


class MusicDataset(Dataset):
    # Initialising the
    def __init__(self, data_dir, metadata_file, labels_dir, max_size_bytes=None):
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_file)
        self.labels_dir = labels_dir
        self.filenames = [filename for filename in os.listdir(data_dir) if filename.endswith('.wav')]
        if max_size_bytes is not None:
            total_size = 0
            new_filenames = []
            for filename in self.filenames:
                file_path = os.path.join(data_dir, filename)
                file_size = os.path.getsize(file_path)
                if total_size + file_size <= max_size_bytes:
                    new_filenames.append(filename)
                    total_size += file_size
                else:
                    break
            self.filenames = new_filenames

    def __len__(self):
        return len(self.filenames)

    def _cut_if_necessary(self, signal):
        # signal -> (1, num_sample)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        file_path = os.path.join(self.data_dir, filename)
        audio, sr = librosa.load(file_path, sr=None)

        # Extract metadata for the current filename
        metadata_row = self.metadata[self.metadata['id'] == filename[:-4]]  # Removing '.wav' extension

        if metadata_row.empty:
            print(f"No metadata found for filename: {filename}")
            return

        composer = metadata_row['composer'].values[0]
        composition = metadata_row['composition'].values[0]
        movement = metadata_row['movement'].values[0]
        ensemble = metadata_row['ensemble'].values[0]

        # Load labels for the current filename
        labels_file = os.path.join(self.labels_dir, filename[:-4] + '.csv')  # Removing '.wav' extension
        labels_data = pd.read_csv(labels_file)

        # You can process labels_data further as needed

        # You might want to perform additional preprocessing here

        return audio, composer, composition, movement, ensemble, labels_data

    def _right_pad_if_necessary(self, signal):
        len_signal = signal.shape[1]
        if len_signal < self.num_samples:  # apply right pad
            num_missing_samples = self.num_samples - len_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        # signal = (channels, num_samples) -> (2, 16000) -> (1, 16000)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, item):
        fold = f"fold{self.annotations.iloc[item, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[item, 0])
        return path


# Example usage:
project_dir = os.getcwd()
train_data_dir = os.path.join(project_dir, 'Data', 'musicnet', 'musicnet', 'train_data')
metadata_file = os.path.join(project_dir, 'Data', 'musicnet_metadata.csv')
labels_dir = os.path.join(project_dir, 'Data', 'musicnet', 'musicnet', 'train_labels')
max_train_data_size = 1 * 1024 * 1024 * 1024  # 1GB in bytes
train_dataset = MusicDataset(train_data_dir, metadata_file, labels_dir, max_size_bytes=max_train_data_size)

# Create a DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Iterate over the DataLoader
"""for batch in train_loader:
    audio_batch, composer_batch, composition_batch, movement_batch, ensemble_batch, labels_batch = batch
    # Do something with the batch of audio data, metadata, and labels"""
