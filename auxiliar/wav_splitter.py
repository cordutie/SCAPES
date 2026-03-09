import os
import soundfile as sf

def split_wavs_into_train_validation(folder_path, train_percentage):
    """
    Splits all WAV files in a given folder into training and validation parts.

    Args:
        folder_path (str): The path to the folder containing WAV files.
        train_percentage (float): The percentage of each file to be used for the training split (e.g., 0.8 for 80%).
                                  The remaining (1 - train_percentage) will be for validation.
    """
    if not (0 < train_percentage < 1):
        print("Error: train_percentage must be between 0 and 1 (exclusive).")
        return

    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return

    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

    if not wav_files:
        print(f"No WAV files found in '{folder_path}'.")
        return

    print(f"Found {len(wav_files)} WAV files in '{folder_path}'.")

    for filename in wav_files:
        file_path = os.path.join(folder_path, filename)
        try:
            # Read the audio file
            data, samplerate = sf.read(file_path)

            # Calculate split point
            num_samples = len(data)
            train_samples = int(num_samples * train_percentage)

            # Split data
            train_data = data[:train_samples]
            validation_data = data[train_samples:]

            # Construct new filenames
            base_name, ext = os.path.splitext(filename)
            train_filename = f"{base_name}_train{ext}"
            validation_filename = f"{base_name}_validation{ext}"

            train_file_path = os.path.join(folder_path, train_filename)
            validation_file_path = os.path.join(folder_path, validation_filename)

            # Write the new files
            sf.write(train_file_path, train_data, samplerate)
            sf.write(validation_file_path, validation_data, samplerate)

            print(f"  - Split '{filename}' into '{train_filename}' and '{validation_filename}'.")

        except Exception as e:
            print(f"  - Error processing '{filename}': {e}")

    print("WAV file splitting complete.")

# Example usage (you can uncomment and modify this to test):
# # 1. Create a dummy folder and some dummy wav files for testing
# os.makedirs('test_wavs', exist_ok=True)
# import numpy as np
# for i in range(3):
#     dummy_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 5, 44100 * 5)).astype(np.float32)
#     sf.write(f'test_wavs/dummy_audio_{i}.wav', dummy_signal, 44100)

# # 2. Call the function
# split_wavs_into_train_validation(folder_path='test_wavs', train_percentage=0.7)

# # 3. Check the contents of the folder
# print("\nFiles in 'test_wavs' after splitting:")
# print(os.listdir('test_wavs'))