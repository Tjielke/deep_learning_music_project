import os
import librosa
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class MusicDataset(Dataset):
    def __init__(self, data_dir, dataframe,transformation,
                 target_sample_rate, num_samples, device, max_size_bytes=None):
        self.data_dir = data_dir
        self.filenames = [filename for filename in os.listdir(data_dir) if filename.endswith('.wav')]
        self.dataframe = dataframe
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
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

    def _get_audio_sample_label(self, item):
        return self.dataframe.iloc[item, 2]

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        file_path = os.path.join(self.data_dir, filename)
        label = self._get_audio_sample_label(idx)
        signal, sr = torchaudio.load(file_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)

        return signal, label


