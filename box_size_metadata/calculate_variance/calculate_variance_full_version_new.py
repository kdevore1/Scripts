# calculate_variance.py
# The purpose of this script is to read in a list of .mrc files calculate the background variances, signal variances, and SNRs. 
# These values will then be written to a .csv file.

########################################################################

# Imports
import numpy as np
import mrcfile
import csv
import glob
import time
import pandas as pd

# start time
st = time.time()

# reading in files 
file_name_1_0 = glob.glob('*_1.0.mrc')
file_name_1_5 = glob.glob('*_1.5.mrc')
file_name_2_0 = glob.glob('*_2.0.mrc')

########################################################################

# Calculating background & signal variances

# diameter of mask
# to calculate take 1/px size and multiply by particle diameter (round to even number) 
diam = 264 # 330

# radius of mask
rad = diam / 2

# list to append background variances to 
background_variances = []

# list to append signal variances to 
signal_variances = []

# loop through each file
for file in file_name_2_0:
    
    # read in file
    file = mrcfile.read(file)

    # selects ONLY micrographs with particles picked
    if len(np.shape(file)) == 3: 

        # calculate the center of the image
        c_rows = (((np.shape(file)[1]) - 1 ) / 2) 
        c_cols = (((np.shape(file)[2]) - 1 ) / 2)

        # create a grid of coordinates
        xx, yy = np.meshgrid(range(np.shape(file)[1]), range(np.shape(file)[2]))

        # calculate the distance from the center of the image for each point
        d = np.sqrt((xx - c_rows)**2 + (yy - c_cols)**2)

        # create numpy arrays of zeros to store the intensity values
        background_data = np.zeros((np.shape(file)[1], np.shape(file)[2]))
        signal_data = np.zeros((np.shape(file)[1], np.shape(file)[2]))

        # z dimension (# of particles in the stack)  
        for dimension in range(np.shape(file)[0]):
            
            # select the intensity values for the background and signal
            background_data[d > rad] = file[dimension, d > rad]
            signal_data[d <= rad] = file[dimension, d <= rad]

        # calculate the variance of the background intensity values
        background_variance = np.var(background_data)
        background_variances.append(background_variance)

        # calculate the variance of the signal intensity values
        signal_variance = np.var(signal_data)
        signal_variances.append(signal_variance)
    

# calculate average variance of data set
avg_background_variance = np.mean(background_variances)
avg_signal_variance = np.mean(signal_variances)
print("Average background variance is: ", avg_background_variance)
print("Average signal variance is: ", avg_signal_variance)

########################################################################

# Calculating signal to noise ratio

SNRs = []

for signal, background in zip(signal_variances, background_variances):
    SNR = signal / background
    SNRs.append(SNR)

########################################################################

# Creating output csv file

# header for csv
header_list = ['Signal Variance', 'Background Variance', 'SNR']

# zipping list, writing to data frame, writing to csv
combined_variance_list = zip(signal_variances, background_variances, SNRs) 
df = pd.DataFrame(combined_variance_list, columns = ['signal_variances', 'background_variances', 'SNRs'])
df.to_csv('avg_variance_2_0.csv', index = False, header = header_list)

########################################################################

# Calculate run time

# end time
et = time.time()

# execution time (hours)
elapsed_time = et - st
elapsed_time_mins = elapsed_time / 60
print('Execution time:', elapsed_time_mins, 'mins')