import os
import librosa
import torch
from torch.utils.data import Dataset, DataLoader

class MusicDataset(Dataset):
    def __init__(self, data_dir, max_size_bytes=None):
        self.data_dir = data_dir
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

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        file_path = os.path.join(self.data_dir, filename)
        audio, sr = librosa.load(file_path, sr=None)
        # You might want to perform additional preprocessing here
        return audio, filename  # Assuming labels are the filenames in this case


