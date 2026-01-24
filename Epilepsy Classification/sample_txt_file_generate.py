import numpy as np

sample_length = 4097  # same as Bonn dataset
output_file = "sample_patient.txt"

eeg_data = np.random.normal(loc=0.0, scale=50.0, size=sample_length)  # mean=0, std=50
np.savetxt(output_file, eeg_data, fmt="%.2f")  # 2 decimal places

print(f"Sample EEG file saved as: {output_file}")