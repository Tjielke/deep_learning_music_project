import librosa
import librosa.display
import pandas as pd
import numpy as np


your_path = 'C:/Users/20182877/JADS C schijf/Year_2/Semester_2/Deep Learning/Data/'

list_of_csv_paths = [your_path + 'onsets_ISMIR_2012/onsets_ISMIR_2012/annotations/onsets/ah_development_percussion_castagnet1.onsets',
                     your_path + '']

list_of_wav_paths = [your_path + 'onsets_ISMIR_2012/onsets_ISMIR_2012/audio/ah_development_percussion_castagnet1.flac',
                     your_path + '',]

n_fft = 64
hop_length = 64
number_mels = 254


def compute_and_save_spect_target_dataframes(list_of_csv_paths, list_of_wav_paths, your_path,hop_length,n_fft,number_mels):
    """
    Computes start_end_spect_target dataframes for each CSV and WAV file path in the provided lists
    and saves them as CSV files in the specified location.

    Parameters:
    - list_of_csv_paths (list): A list of file paths to the CSV files containing the annotations.
    - list_of_wav_paths (list): A list of file paths to the WAV audio files.
    - your_path (str): The path where the dataframes will be saved as CSV files.

    Returns:
    - list of str: A list of messages confirming the successful saving of each file.

    Raises:
    - ValueError: If the lengths of the CSV and WAV file path lists do not match.
    """
    # Ensure the lengths of the lists are the same
    if len(list_of_csv_paths) != len(list_of_wav_paths):
        raise ValueError("The lengths of the CSV and WAV file path lists must be equal.")

    # Initialize a list to hold confirmation messages
    confirmation_messages = []

    # Iterate over each pair of CSV and WAV file paths
    for csv_path, wav_path in zip(list_of_csv_paths, list_of_wav_paths):
        # Extract the name of the file (e.g., 'Haslebuskane_happy' from the WAV file path)
        file_name = wav_path.split('/')[-1].replace('.flac', '')

        # Load the annotations from the CSV file
        label_frame = pd.read_csv(csv_path, header=None)
        onset_list = label_frame[0].values.tolist()

        # Load the audio file
        y, sr = librosa.load(wav_path)

        # Compute the Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=number_mels)
        num_frames = mel_spectrogram.shape[1]
        
        # Initialize a DataFrame to hold the start, end samples, spectrogram, and onset data
        start_end_spect_target = pd.DataFrame(columns=['Start sample', 'End sample', 'Spectrogram', 'onset'])

        # Calculate sample time
        frame_duration = hop_length / sr
        
        if n_fft == hop_length:
            # Case where n_fft equals hop_length
            start_samples = np.arange(num_frames) * frame_duration
            end_samples = start_samples + frame_duration
            
            start_end_spect_target = pd.DataFrame({
                'Start sample': start_samples,
                'End sample': end_samples,
                'Spectogram': [mel_spectrogram[:, i] for i in range(num_frames)],
                'onset': 0
            })
        else:
            # Case where n_fft is not equal to hop_length
            start_samples = np.arange(1, num_frames) * frame_duration
            end_samples = start_samples + frame_duration
            
            diffs = [mel_spectrogram[:, i - 1] - mel_spectrogram[:, i] for i in range(1, num_frames)]
            
            start_end_spect_target = pd.DataFrame({
                'Start sample': start_samples,
                'End sample': end_samples,
                'Spectogram': diffs,
                'onset': 0
            })
    
        # Mark the onset frames based on the onset list
        for onset in onset_list:
            mask = (start_end_spect_target['Start sample'] <= onset) & (onset <= start_end_spect_target['End sample'])
            start_end_spect_target.loc[mask, 'onset'] = 1
                # Save the computed start_end_spect_target DataFrame as a CSV file
        csv_save_path = f"{your_path}/{file_name}_start_end_spect_target.csv"
        start_end_spect_target.to_csv(csv_save_path, index=False)

        # Add a confirmation message to the list
        confirmation_messages.append(f"File '{csv_save_path}' saved successfully.")

    return confirmation_messages


compute_and_save_spect_target_dataframes(list_of_csv_paths, list_of_wav_paths, your_path,n_fft,hop_length,number_mels)
