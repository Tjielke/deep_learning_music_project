import librosa
import librosa.display
import pandas as pd
import numpy as np
import os

base_path = os.getcwd()

annotations_dir = os.path.join(base_path, 'onsets_ISMIR_2012', 'annotations', 'onsets')
audio_dir = os.path.join(base_path, 'onsets_ISMIR_2012', 'audio')

# Retrieve all csv and audio files
list_of_onset_paths = [os.path.join(annotations_dir, f) for f in os.listdir(annotations_dir) if f.endswith('.onsets')]
list_of_flac_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.flac')]

n_fft = 64
hop_length = 64
number_mels = 254


def compute_and_save_spect_target_dataframes(list_of_csv_paths, list_of_wav_paths, your_path, hop_length, n_fft,
                                             number_mels):
    # Ensure the lengths of the lists are the same
    if len(list_of_csv_paths) != len(list_of_wav_paths):
        raise ValueError("The lengths of the CSV and WAV file path lists must be equal.")

    # Initialize a list to hold confirmation messages
    confirmation_messages = []

    # Iterate over each pair of CSV and WAV file paths
    for csv_path, wav_path in zip(list_of_csv_paths, list_of_wav_paths):
        # Extract the name of the file (e.g., 'Haslebuskane_happy' from the WAV file path)
        file_name = wav_path.split('/')[-1].replace('.flac', '')

        # Extract the name of the file without extension
        name = os.path.splitext(os.path.basename(wav_path))[0]

        # Load the annotations from the CSV file
        label_frame = pd.read_csv(csv_path, header=None)
        onset_list = label_frame[0].values.tolist()

        # Load the audio file
        y, sr = librosa.load(wav_path)
        
        

        # Compute the Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                                         n_mels=number_mels)
        num_frames = mel_spectrogram.shape[1]

        # Initialize a DataFrame to hold the start, end samples, spectrogram, and onset data
        start_end_spect_target = pd.DataFrame(columns=['Start sample', 'End sample', 'Spectrogram', 'onset'])

        # Calculate sample time
        frame_duration = hop_length / sr

        # Prepare data frames for storing the spectrogram information
        start_samples = np.arange(num_frames) * frame_duration
        end_samples = start_samples + frame_duration

        # Create DataFrame
        start_end_spect_target = pd.DataFrame({
            'Start sample': start_samples,
            'End sample': end_samples,
            'Spectrogram': [mel_spectrogram[:, i] for i in range(num_frames)],
            'onset': 0
        })

        # Mark the onset frames based on the onset list
        for onset in onset_list:
            mask = (start_end_spect_target['Start sample'] <= onset) & (onset <= start_end_spect_target['End sample'])
            start_end_spect_target.loc[mask, 'onset'] = 1

        # Construct the path where each CSV will be saved
        csv_save_path = os.path.join(your_path, 'onsets_ISMIR_2012', 'new_csv_files',
                                     f"{name}_start_end_spect_target.csv")

        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
        print(csv_save_path)
        # Save the computed DataFrame as a CSV file
        #start_end_spect_target.to_csv(csv_save_path, index=False)

        # Add a confirmation message to the list
        confirmation_messages.append(f"File '{csv_save_path}' saved successfully.")

    return confirmation_messages


compute_and_save_spect_target_dataframes(list_of_onset_paths, list_of_flac_files, base_path,n_fft,hop_length,number_mels)
