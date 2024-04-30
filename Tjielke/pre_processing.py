import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

##Import file
your_path = 'C:/Users/20182877/JADS C schijf/Year_2/Semester_2/Deep Learning/Data/'

list_of_csv_paths = [your_path + 'HF1/HF1/Songs/Haslebuskane/Aligned annotations/Ground truth/Haslebuskane_happy.csv',
                     your_path + 'HF1/HF1/Songs/Haslebuskane/Aligned annotations/Ground truth/Haslebuskane_angry.csv',
                     your_path + 'HF1/HF1/Songs/Haslebuskane/Aligned annotations/Ground truth/Haslebuskane_sad.csv',
                     your_path + 'HF1/HF1/Songs/Haslebuskane/Aligned annotations/Ground truth/Haslebuskane_tender.csv',
                     your_path + 'HF1/HF1/Songs/Haslebuskane/Haslebuskane_original 10-Jul-2020 12-07-28']

list_of_wav_paths = [your_path + 'HF1/HF1/Songs/Haslebuskane/Audio files/Haslebuskane_happy.wav',
                     your_path + 'HF1/HF1/Songs/Haslebuskane/Audio files/Haslebuskane_angry.wav',
                     your_path + 'HF1/HF1/Songs/Haslebuskane/Audio files/Haslebuskane_sad.wav',
                     your_path + 'HF1/HF1/Songs/Haslebuskane/Audio files/Haslebuskane_tender.wav',
                     your_path + 'HF1/HF1/Songs/Haslebuskane/Audio files/Haslebuskane_original.wav']


def compute_onset_lists(list_of_csv_paths, list_of_wav_paths):
    """
    Computes onset lists for each CSV and WAV file path in the provided lists.

    Parameters:
    - list_of_csv_paths (list): A list of file paths to the CSV files containing the annotations.
    - list_of_wav_paths (list): A list of file paths to the WAV audio files.

    Returns:
    - dict: A dictionary mapping each file name (after the last slash in the WAV file path) to its corresponding onset list.

    Raises:
    - ValueError: If the lengths of the CSV and WAV file path lists do not match.
    """
    # Ensure the lengths of the lists are the same
    if len(list_of_csv_paths) != len(list_of_wav_paths):
        raise ValueError("The lengths of the CSV and WAV file path lists must be equal.")

    onset_lists = {}

    # Iterate over each pair of CSV and WAV file paths
    for csv_path, wav_path in zip(list_of_csv_paths, list_of_wav_paths):
        # Extract the name of the file (e.g., 'Haslebuskane_happy' from the WAV file path)
        file_name = wav_path.split('/')[-1].replace('.wav', '')

        # Load the annotations from the CSV file
        label_frame = pd.read_csv(csv_path, header=None)
        onset_list = label_frame[0]

        # Load the audio file
        y, sr = librosa.load(wav_path)

        # Compute the Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=64, hop_length=64, n_mels=254)

        # # Convert the Mel spectrogram to decibel scale (optional)
        # mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        #
        # # (Optional) Plot the Mel spectrogram
        # plt.figure(figsize=(10, 4))
        # librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
        # plt.colorbar(format='%+2.0f dB')
        # plt.title('Mel Spectrogram')
        # plt.show()

        # Initialize a DataFrame to hold the start, end samples, spectrogram, and onset data
        start_end_spect_target = pd.DataFrame(columns=['Start sample', 'End sample', 'Spectrogram', 'onset'])

        # Calculate sample time
        sample_time = (1 / sr) * 64

        # Fill the DataFrame with start and end samples, spectrogram, and default onset value of 0
        for i in range(mel_spectrogram.shape[1]):
            start_end_spect_target.loc[len(start_end_spect_target.index)] = [(0 + i * sample_time),
                                                                             (sample_time + i * sample_time),
                                                                             mel_spectrogram[:, i], 0]

        # Update onset values based on the provided onset list
        for onset in onset_list:
            # Find rows where onset falls between start sample and end sample
            mask = (start_end_spect_target['Start sample'] <= onset) & (onset <= start_end_spect_target['End sample'])
            # Update 'onset' column to 1 where mask is True
            start_end_spect_target.loc[mask, 'onset'] = 1

        # Add the computed onset list to the dictionary using the file name as the key
        onset_lists[file_name] = onset_list

    return onset_lists


