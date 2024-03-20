# chromatogram_plotter.py last updated 19 March 2024 by Kira DeVore
# The purpose of this script is to allow a user to plot SEC data in the format of a .txt file(s) into a legible figure.


# Imports
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from scipy.signal import find_peaks
import os
import chardet

###########################################################################################

# Functions
# Function to read data from a text file and append to a 2D array
def read_and_append_data(file):
    # Detect the encoding of the file
    with open(file, 'rb') as f:
        result = chardet.detect(f.read())

    # Reading in text file as dataframe
    df = pd.read_csv(file, encoding = result['encoding'], sep = '\t')

    # Extracting volume and mAU columns and converting to numpy arrays - Ignore header info
    raw_vol = df['Chrom.1'].to_numpy()[2:]  
    raw_mAU = df['Unnamed: 1'].to_numpy()[2:]

    # Convert all string elements to numerics
    raw_vol = pd.to_numeric(raw_vol, errors = 'coerce')
    raw_mAU = pd.to_numeric(raw_mAU, errors = 'coerce')

    return raw_vol, raw_mAU

###########################################################################################

# Truncation
def Truncate_Data(raw_vol_data_array, raw_mAU_data_array, lower_threshold, upper_threshold):
    truncated_vol_data_array = []
    truncated_mAU_data_array = []
    truncated_indices = []

    max_length = 0
    
    for dataset_idx in range(raw_vol_data_array.shape[0]):
        dataset_vol = raw_vol_data_array[dataset_idx]
        dataset_mAU = raw_mAU_data_array[dataset_idx]

        # Find indices of vol values within the specified range
        indices_to_keep = np.where((dataset_vol >= lower_threshold) & (dataset_vol <= upper_threshold))[0]

        # Save indices of truncated dataset
        truncated_indices.append(indices_to_keep)

        # Truncate dataset using the indices
        truncated_vol = dataset_vol[indices_to_keep]
        truncated_mAU = dataset_mAU[indices_to_keep]

        # Update max_length if necessary
        max_length = max(max_length, len(truncated_vol))

        # Append truncated dataset to the list
        truncated_vol_data_array.append(truncated_vol)
        truncated_mAU_data_array.append(truncated_mAU)

    # Pad shorter arrays with NaN values to match max_length
    for i in range(len(truncated_vol_data_array)):
        truncated_vol_data_array[i] = np.pad(truncated_vol_data_array[i], (0, max_length - len(truncated_vol_data_array[i])), mode = 'constant', constant_values = np.nan)
        truncated_mAU_data_array[i] = np.pad(truncated_mAU_data_array[i], (0, max_length - len(truncated_mAU_data_array[i])), mode = 'constant', constant_values = np.nan)

    # Convert lists to numpy arrays
    truncated_vol_data_array = np.array(truncated_vol_data_array)
    truncated_mAU_data_array = np.array(truncated_mAU_data_array)

    return truncated_vol_data_array, truncated_mAU_data_array, truncated_indices

###########################################################################################

# Fourier Filtering
def Fourier_filter(truncated_vol_data_array, truncated_mAU_data_array):
    filtered_mAU_data_array = np.zeros_like(truncated_mAU_data_array)
    filtered_vol_data_array = np.zeros_like(truncated_vol_data_array)
    
    for i in range(total_num_data_sets):
        
        # Setting nan values to 0
        no_nan_mAU = np.nan_to_num(truncated_mAU_data_array[i], copy = True, nan = 0.0)
        no_nan_vol = np.nan_to_num(truncated_vol_data_array[i], copy = True, nan = 0.0)

        # Apply FFT
        fft_array = np.fft.fft(no_nan_mAU)

        # Define low-pass filter - cutoff frequency (Hz)
        fc = 2

        # Calculates the frequencies associated with the Fourier transform of a signal
        freqs = np.fft.fftfreq(len(fft_array), no_nan_vol[1] - no_nan_vol[0])

        # Gaussian filter - filters out certain frequencies
        mask = np.exp(-0.5 * (freqs / fc) ** 2)

        # Threshold filter - frequencies below 20% of the cut off are kept
        mask[np.abs(freqs) < fc / 10] = 1

        # Applies masked filter to Fourier transformed signal
        fft_array_filtered = fft_array * mask

        # Apply inverse FFT
        ifft_array = np.real(np.fft.ifft(fft_array_filtered))

        # Append values to each row
        filtered_mAU_data_array[i, :] = ifft_array 
        filtered_vol_data_array[i, :] = no_nan_vol 
        
        # Replace zeros with nan to fix plotting error
        filtered_mAU_data_array[filtered_mAU_data_array == 0] = np.nan
        filtered_vol_data_array[filtered_vol_data_array == 0] = np.nan
        
    return filtered_vol_data_array, filtered_mAU_data_array

###########################################################################################

# Alignment
def Peak_Alignment(filtered_vol_data_array, filtered_mAU_data_array, num_peaks):
    
    # Reference data set
    ref_data_set_number = int(input("Which data set is the reference? ")) - 1

    # Create a 3D array to hold index, vol, mAU
    maxima_3d_array = np.zeros((len(filtered_vol_data_array), 3, num_peaks))

    # Scaling the volume values so that the peaks appear more aligned with the reference 
    for i in range(len(filtered_mAU_data_array)):
        # Printing data set number
        print("Data set: ", i + 1)

        # For reference data set...
        if i == ref_data_set_number:
            # Finding peaks in reference
            ref_mAU_peaks_index, ref_mAU_peaks = find_peaks(filtered_mAU_data_array[i], height = 0.1) 
            ref_mAU_peaks = ref_mAU_peaks['peak_heights']

            # Printing indices of peaks
            print("Ref mAU index:", ref_mAU_peaks_index)

            # Printing actual peak values
            print("Ref mAU:", ref_mAU_peaks)

            # Locating corresponding volume values of peaks
            ref_vol_peaks = filtered_vol_data_array[i][ref_mAU_peaks_index] 

            # Printing volume peak values
            print("Ref vol:", ref_vol_peaks)

            # Create an empty list to store user decisions for reference data set
            user_decisions = []

            # Iterate over each value in the first array for reference data set
            for idx, value in enumerate(ref_vol_peaks):
                # Prompt the user
                user_input = input(f"Do you want to keep the {value} value? (y/n): ")

                # Check user input
                if user_input == 'y':
                    user_decisions.append(True)
                elif user_input == 'n':
                    user_decisions.append(False)
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")

            # Convert the list to a boolean NumPy array
            keep_values = np.array(user_decisions)

            # Use boolean indexing to filter values in corresponding arrays for reference data set
            ref_mAU_peaks_index = ref_mAU_peaks_index[keep_values]
            maxima_3d_array[i, 0, :] = ref_mAU_peaks_index
            ref_vol_peaks = ref_vol_peaks[keep_values]
            maxima_3d_array[i, 1, :] = ref_vol_peaks
            ref_mAU_peaks = ref_mAU_peaks[keep_values]
            maxima_3d_array[i, 2, :] = ref_mAU_peaks

        # Run through the normal steps and append to 3d array
        else:
            # Find peak values
            mAU_peaks_index, mAU_peaks = find_peaks(filtered_mAU_data_array[i], height = 0.1) 
            mAU_peaks = mAU_peaks['peak_heights']

            # Printing indices of peaks
            print("mAU index:", mAU_peaks_index)

            # Printing actual peak values
            print("mAU:", mAU_peaks)

            # Locating corresponding volume values of peaks
            vol_peaks = filtered_vol_data_array[i][mAU_peaks_index] 

            # Printing volume peak values
            print("Vol:", vol_peaks)

            # Create an empty list to store user decisions for reference data set
            user_decisions = []

            # Iterate over each value in the first array for data set
            for idx, value in enumerate(vol_peaks):
                # Prompt the user
                user_input = input(f"Do you want to keep the value at {value}? (y/n): ")

                # Check user input
                if user_input == 'y':
                    user_decisions.append(True)
                elif user_input == 'n':
                    user_decisions.append(False)
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")

            if len(mAU_peaks_index) == 1:
                # Ask the user which peak to align with (first or second peak of the reference dataset)
                alignment_choice = input("Dataset has a single peak. Align with the first or second peak of the reference? ")

                if alignment_choice == '1':
                    # Align with the first peak of the reference dataset
                    maxima_3d_array[i, 0, 0] = mAU_peaks_index[0]
                    maxima_3d_array[i, 1, 0] = vol_peaks[0]
                    maxima_3d_array[i, 2, 0] = mAU_peaks[0]
                elif alignment_choice == '2':
                    # Align with the second peak of the reference dataset
                    ref_peak_index = maxima_3d_array[ref_data_set_number, 0, 1] 
                    closest_peak_index = np.argmin(np.abs(mAU_peaks_index - ref_peak_index))
                    maxima_3d_array[i, 0, 1] = mAU_peaks_index[closest_peak_index]
                    maxima_3d_array[i, 1, 1] = vol_peaks[closest_peak_index]
                    maxima_3d_array[i, 2, 1] = mAU_peaks[closest_peak_index]
                else:
                    print("Invalid input. Please enter '1' or '2'.")
            else:
                # Convert the list to a boolean NumPy array
                keep_values = np.array(user_decisions)

                # Use boolean indexing to filter values in corresponding arrays for data set
                mAU_peaks_index = mAU_peaks_index[keep_values]
                maxima_3d_array[i, 0, :] = mAU_peaks_index
                vol_peaks = vol_peaks[keep_values]
                maxima_3d_array[i, 1, :] = vol_peaks
                mAU_peaks = mAU_peaks[keep_values]
                maxima_3d_array[i, 2, :] = mAU_peaks

    # Array to store alphas
    alphas = np.zeros(len(filtered_vol_data_array))

    # Blank array to store alpha volume values 
    alpha_vol_data_array = np.zeros_like(filtered_vol_data_array)

    # Scaling x-axis
    for i in range(len(filtered_vol_data_array)):
        # Check if there are any zeros in the current layer of maxima_3d_array_with_nans
        if np.any(maxima_3d_array[i] == 0):
            # If zeros are present, use only the second peak in ref_vol_peaks
            alphas[i] = ref_vol_peaks[1] / maxima_3d_array[i, 1, 1]
        else:
            # Calculate alpha and append to alphas array
            alphas[i] = np.sum(ref_vol_peaks / maxima_3d_array[i, 1, :]) / num_peaks

        # Multiplying alpha by the volume values
        alpha_vol_data_array[i] = filtered_vol_data_array[i] * alphas[i]

    return maxima_3d_array, alphas, alpha_vol_data_array, filtered_mAU_data_array


###########################################################################################

# Maximization Normalization
def Maximization_Normalization(mAU_data_array):
    normalized_mAU_data_array = np.zeros_like(mAU_data_array)

    # For each dataset's mAU...
    for i in range(mAU_data_array.shape[0]):
        dataset_mAU = mAU_data_array[i]

        # Calculate the maximum value
        max = np.max(dataset_mAU)

        # Normalize the dataset by dividing by the maximum value
        normalized_mAU = dataset_mAU / max

        # Store the normalized dataset
        normalized_mAU_data_array[i, :] = normalized_mAU

    return normalized_mAU_data_array


###########################################################################################

# AUC Normalization
def AUC_Normalization(mAU_data_array):

    normalized_mAU_data_array = np.zeros_like(mAU_data_array)

    # For each dataset's mAU...
    for i in range(mAU_data_array.shape[0]):
        dataset_mAU = mAU_data_array[i]

        # Calculate the area under the curve
        AUC = np.trapz(dataset_mAU)

        # Normalize the dataset by dividing by the AUC
        normalized_mAU = dataset_mAU / AUC

        # Store the normalized dataset
        normalized_mAU_data_array[i, :] = normalized_mAU

        print(normalized_mAU_data_array.shape)

    return normalized_mAU_data_array


###########################################################################################

# Plotting function with customizable volume and mAU arrays
def Plot(vol_array, mAU_array):
    # All fonts Arial
    plt.rcParams["font.family"] = "Arial"

    # Plot each dataset
    for i in range(len(vol_array)):
        plt.plot(vol_array[i], mAU_array[i], linewidth = 1)
    
    # X-axis
    plt.xlabel('Volume (mL)')
    # Y-axis
    plt.ylabel('mAU')
    # Legend
    plt.legend()
    # Saves figure
    plt.savefig('plot.png', dpi = 800)
    plt.show()
    print("File saved!")

###########################################################################################


# Read in all .txt files
# Specify the folder path containing the text files
folder_path = '/Users/kiradevore/Documents/python_scripts/chromatogram_plotter/'

# Get a list of all text files in the folder
text_files = glob.glob(os.path.join(folder_path, '*.txt'))

# Sorting text files
text_files.sort()

# If more than one text file is located in the folder...
if len(text_files) > 1:

    # Printing number of text files in the folder
    print(len(text_files), "text files in the folder.")

    # Print the names of all text files in the folder
    for file in text_files:
        print(file)

    # Asking if you want to plot multiple sets of data on the same plot
    multi_data = input("Are you trying to plot more than one data set on the same plot (y/n)? ")

    # Determining how many data sets there are
    total_num_data_sets = int(input("How many total data sets are there? "))
  

else:
   
    # Print the names of all text files in the folder
    for file in text_files:
        print(file)

    # Setting total number of data sets to 1
    total_num_data_sets = 1



# If plotting numerous data sets...
if multi_data == "y":
    
    # Initialize an empty list to hold raw_vol data arrays
    raw_vol_list = []
    raw_mAU_list = []

    # Loop over each text file
    for file in text_files:
        raw_vol, raw_mAU = read_and_append_data(file)

        # Append each file's volume and mAU data to lists
        raw_vol_list.append(raw_vol)
        raw_mAU_list.append(raw_mAU)

    # Find the length of each sublist and get the maximum length
    longest_row = max(len(sublist) for sublist in raw_vol_list)

    # Nan arrays to hold data
    raw_mAU_data_array = np.full((total_num_data_sets, longest_row), np.nan)
    raw_vol_data_array = np.full((total_num_data_sets, longest_row), np.nan)

    # Fill the arrays with values from the sublists
    for i, sublist in enumerate(raw_vol_list):
        raw_vol_data_array[i, :len(sublist)] = sublist

    for i, sublist in enumerate(raw_mAU_list):
        raw_mAU_data_array[i, :len(sublist)] = sublist

###########################################################################################
    
    # Ask user if they want to truncate data
    truncation = input("Would you like to truncate your data (y/n)? ")

    if truncation == "y":

        # Prompts user for upper / lower thresholds
        lower_threshold = float(input("What is the lower threshold? "))
        upper_threshold = float(input("What is the upper threshold? "))
        
        # Outputs from truncation function
        truncated_vol_data_array, truncated_mAU_data_array, truncated_indices = Truncate_Data(raw_vol_data_array, raw_mAU_data_array, lower_threshold, upper_threshold)
        print("Data truncated.")

    else: 

        print("Data will not be truncated.")

###########################################################################################
    
    # Ask user if they want to filter data
    filtration = input("Are you trying to Fourier filter your data (y/n)? ")
    
    if filtration == "y":

        # Outputs from Fourier filter function
        try:
            filtered_vol_data_array, filtered_mAU_data_array = Fourier_filter(truncated_vol_data_array, truncated_mAU_data_array)
            print("Data filtered.")
        except:
            filtered_vol_data_array, filtered_mAU_data_array = Fourier_filter(raw_vol_data_array, raw_mAU_data_array)
            print("Data filtered.")

    else: 

        print("Data will not be filtered.")

###########################################################################################
    
    # Ask user if they want to align peaks 
    alignment = input("Are you trying to align the peaks in your data sets (y/n)? ")

    if alignment == "y":
    
        if 'filtered_vol_data_array' in locals() and 'filtered_mAU_data_array' in locals() and \
        filtered_vol_data_array is not None and filtered_mAU_data_array is not None:
            
            # Number of peaks used for analysis
            num_peaks = int(input("How many peaks do you want to use for the alignment? "))
            maxima_3d_array, alphas, alpha_vol_data_array, filtered_mAU_data_array = Peak_Alignment(filtered_vol_data_array, filtered_mAU_data_array, num_peaks)
            print("Peaks aligned.")
            print("Scaling factors: ", alphas)
        
        else:
            print("You need to filter the data before alignment to remove erroneous peaks")
            exit()

    else: 

        print("Data will not be aligned.")

###########################################################################################
    
    # Ask user if they want to normalize data by any criteria
    normalization = input("Are you trying to normalize your data your data by (1) maximization, (2) area under the curve, (n) or plot as is? ")
    
    if normalization == "1":
        if 'filtered_mAU_data_array' in locals() and filtered_mAU_data_array is not None:
            normalized_mAU_data_array = Maximization_Normalization(filtered_mAU_data_array) 
            print("Filtered data normalized according to the maximum value.")
        elif 'truncated_mAU_data_array' in locals() and truncated_mAU_data_array is not None:
            normalized_mAU_data_array = Maximization_Normalization(truncated_mAU_data_array)
            print("Truncated data normalized according to the maximum value.")
        else:
            normalized_mAU_data_array = Maximization_Normalization(raw_mAU_data_array)
            print("Raw data normalized according to the maximum value.")

    elif normalization == "2":
        if 'filtered_mAU_data_array' in locals() and filtered_mAU_data_array is not None:
            normalized_mAU_data_array = AUC_Normalization(filtered_mAU_data_array) 
            print("Filtered data normalized according to the area under the curve (AUC).")
        elif 'truncated_mAU_data_array' in locals() and truncated_mAU_data_array is not None:
            normalized_mAU_data_array = AUC_Normalization(truncated_mAU_data_array)
            print("Truncated data normalized according to the area under the curve (AUC).")
        else:
            normalized_mAU_data_array = AUC_Normalization(raw_mAU_data_array)
            print("Raw data normalized according to the area under the curve (AUC).")
    
    else:
        print("Data will not be normalized.")
    
########################################################################################### 
    
    # Plotting 
    if 'alpha_vol_data_array' in locals() and 'normalized_mAU_data_array' in locals() and \
    alpha_vol_data_array is not None and normalized_mAU_data_array is not None:
        print("Plotting aligned and normalized data.")
        Plot(alpha_vol_data_array, normalized_mAU_data_array)
    elif 'alpha_vol_data_array' in locals() and 'filtered_mAU_data_array' in locals() and \
    alpha_vol_data_array is not None and filtered_mAU_data_array is not None:
        print("Plotting aligned and filtered data.")
        Plot(alpha_vol_data_array, filtered_mAU_data_array)
    elif 'filtered_vol_data_array' in locals() and 'normalized_mAU_data_array' in locals() and \
    filtered_vol_data_array is not None and normalized_mAU_data_array is not None:
        print("Plotting filtered and normalized data.")
        Plot(filtered_vol_data_array, normalized_mAU_data_array)
    elif 'filtered_vol_data_array' in locals() and 'filtered_mAU_data_array' in locals() and \
        filtered_vol_data_array is not None and filtered_mAU_data_array is not None:
        print("Plotting filtered data.")
        Plot(filtered_vol_data_array, filtered_mAU_data_array)
    elif 'truncated_vol_data_array' in locals() and 'normalized_mAU_data_array' in locals() and \
    truncated_vol_data_array is not None and normalized_mAU_data_array is not None:
        print("Plotting truncated and normalized data.")
        Plot(truncated_vol_data_array, normalized_mAU_data_array)
    elif 'truncated_vol_data_array' in locals() and 'filtered_mAU_data_array' in locals() and \
    truncated_vol_data_array is not None and filtered_mAU_data_array is not None:
        print("Plotting truncated and filtered data.")
        Plot(truncated_vol_data_array, filtered_mAU_data_array)
    elif 'truncated_vol_data_array' in locals() and 'truncated_mAU_data_array' in locals() and \
    truncated_vol_data_array is not None and truncated_mAU_data_array is not None:
        print("Plotting truncated and filtered data.")
        Plot(truncated_vol_data_array, truncated_mAU_data_array)
    else:
        print("Plotting raw data.")
        Plot(raw_vol_data_array, raw_mAU_data_array)

########################################################################################### 

# If only one data set to plot...
else:
    
    # Convert volume and mAU to Numpy arrays
    raw_vol_data_array, raw_mAU_data_array = read_and_append_data(text_files[0])

########################################################################################### 

    # Ask user if they want to truncate data
    truncation = input("Would you like to truncate your data (y/n)? ")

    if truncation == "y":

        # Prompts user for upper / lower thresholds
        lower_threshold = float(input("What is the lower threshold? "))
        upper_threshold = float(input("What is the upper threshold? "))
        
        # If one dimensional data... reshape is required
        if raw_vol_data_array.ndim == 1:

            # Reshape raw_vol_data_array to have shape (num_datasets, number of columns) - assuming you have only one dataset
            raw_vol_data_array = np.reshape(raw_vol_data_array, (1, -1))  
            raw_mAU_data_array = np.reshape(raw_mAU_data_array, (1, -1))
        
        # Outputs from truncation function
        truncated_vol_data_array, truncated_mAU_data_array, truncated_indices = Truncate_Data(raw_vol_data_array, raw_mAU_data_array, lower_threshold, upper_threshold)
        print("Data truncated.")

    else:
        # If one dimensional data... reshape is required
        if raw_vol_data_array.ndim == 1:

            # Reshape raw_vol_data_array to have shape (num_datasets, number of columns) - assuming you have only one dataset
            raw_vol_data_array = np.reshape(raw_vol_data_array, (1, -1))  
            raw_mAU_data_array = np.reshape(raw_mAU_data_array, (1, -1))
        
        # Outputs from truncation function
        truncated_vol_data_array, truncated_mAU_data_array, truncated_indices = Truncate_Data(raw_vol_data_array, raw_mAU_data_array, 0, 30)
        print("Data will not be truncated.")

###########################################################################################
    
    # Ask user if they want to filter data
    filtration = input("Are you trying to Fourier filter your data (y/n)? ")
    
    if filtration == "y":
        
        try:
            # Outputs from Fourier filter function
            filtered_vol_data_array, filtered_mAU_data_array = Fourier_filter(truncated_vol_data_array, truncated_mAU_data_array)
            print("Data filtered.")
        
        except IndexError:
            print("Data must be truncated in order to be filtered.")

    else: 
        print("Data will not be filtered.")

########################################################################################### 
  
    # Ask user if they want to normalize data by any criteria 
    normalization = input("Are you trying to normalize your data your data by (1) maximization, (2) area under the curve, (n) or plot as is? ")

    if normalization == "1":
        if 'filtered_mAU_data_array' in locals() and filtered_mAU_data_array is not None:
            normalized_mAU_data_array = Maximization_Normalization(filtered_mAU_data_array) 
            print("Filtered data normalized according to the maximum value.")
        elif 'truncated_mAU_data_array' in locals() and truncated_mAU_data_array is not None:
            normalized_mAU_data_array = Maximization_Normalization(truncated_mAU_data_array)
            print("Truncated data normalized according to the maximum value.")
        else:
            normalized_mAU_data_array = Maximization_Normalization(raw_mAU_data_array)
            print("Raw data normalized according to the maximum value.")

    elif normalization == "2":
        if 'filtered_mAU_data_array' in locals() and filtered_mAU_data_array is not None:
            normalized_mAU_data_array = AUC_Normalization(filtered_mAU_data_array) 
            print("Filtered data normalized according to the area under the curve (AUC).")
        elif 'truncated_mAU_data_array' in locals() and truncated_mAU_data_array is not None:
            normalized_mAU_data_array = AUC_Normalization(truncated_mAU_data_array)
            print("Truncated data normalized according to the area under the curve (AUC).")
        else:
            normalized_mAU_data_array = AUC_Normalization(raw_mAU_data_array)
            print("Raw data normalized according to the area under the curve (AUC).")

    else:
        print("Data will not be normalized.")


########################################################################################### 

    # Plotting 
    if 'filtered_vol_data_array' in locals() and 'normalized_mAU_data_array' in locals() and \
    filtered_vol_data_array is not None and normalized_mAU_data_array is not None:
        print("Plotting filtered and normalized data.")
        Plot(filtered_vol_data_array, normalized_mAU_data_array)
    elif 'filtered_vol_data_array' in locals() and 'filtered_mAU_data_array' in locals() and \
        filtered_vol_data_array is not None and filtered_mAU_data_array is not None:
        print("Plotting filtered data.")
        Plot(filtered_vol_data_array, filtered_mAU_data_array)
    elif 'truncated_vol_data_array' in locals() and 'normalized_mAU_data_array' in locals() and \
    truncated_vol_data_array is not None and normalized_mAU_data_array is not None:
        print("Plotting truncated and normalized data.")
        Plot(truncated_vol_data_array, normalized_mAU_data_array)
    elif 'truncated_vol_data_array' in locals() and 'filtered_mAU_data_array' in locals() and \
    truncated_vol_data_array is not None and filtered_mAU_data_array is not None:
        print("Plotting truncated and filtered data.")
        Plot(truncated_vol_data_array, filtered_mAU_data_array)
    elif 'truncated_vol_data_array' in locals() and 'truncated_mAU_data_array' in locals() and \
    truncated_vol_data_array is not None and truncated_mAU_data_array is not None:
        print("Plotting truncated and filtered data.")
        Plot(truncated_vol_data_array, truncated_mAU_data_array)
    else:
        print("Plotting raw data.")
        Plot(raw_vol_data_array, raw_mAU_data_array)