# Imports
import numpy as np
import mrcfile
import csv
import pandas as pd

########################################################################

# File details
# import first micrograph (particle stack)
img_1_0 = mrcfile.read('000016872349906974224_FoilHole_24977990_Data_24016401_24016403_20200225_0402_Fractions.mrc_patch_aligned_doseweighted_particles_1.0.mrc')
img_1_5 = mrcfile.read('000016872349906974224_FoilHole_24977990_Data_24016401_24016403_20200225_0402_Fractions.mrc_patch_aligned_doseweighted_particles_1.5.mrc')
img_2_0 = mrcfile.read('000016872349906974224_FoilHole_24977990_Data_24016401_24016403_20200225_0402_Fractions.mrc_patch_aligned_doseweighted_particles_2.0.mrc')

file_name = [img_1_0, img_1_5, img_2_0]
box_size_names = ['1.0', '1.5', '2.0']

########################################################################

# Calculating background & signal variances

# diameter of mask
# to calculate take 1/px size and multiply by particle diameter (round to even number) 
diam = 264 # 330

# radius of mask
rad = diam / 2

# list to append background intensity values to
background_data = []
background_variances = []

# list to append signal intensity values to
signal_data = []
signal_variances = []

# calculating the variance of the mask and background for each particle in the stack
for file in file_name:   
    # center of image
    c_rows = (((np.shape(file)[1]) - 1 ) / 2) 
    c_cols = (((np.shape(file)[2]) - 1 ) / 2)
    # z dimension (# of particles in the stack)  
    for dimension in range(np.shape(file)[0]):
        # x dimension
        for row in range(np.shape(file)[1]):
            # y dimension
            for column in range(np.shape(file)[2]):
                # distance formula
                d = np.sqrt((row - c_rows)**2 + (column - c_cols)**2)
                # appending the background intensity values 
                if d > rad: 
                    background_data.append(file[dimension][row][column])
                # appending the signal intensity values
                else:
                    signal_data.append(file[dimension][row][column])

    # calculating variance of background intensity values
    background_variance = np.var(background_data)
    background_variances.append(background_variance)

    # calculating variance of signal intensity values
    signal_variance = np.var(signal_data)
    signal_variances.append(signal_variance)

########################################################################

# Calculating signal to noise ratio

SNRs = []

for signal, background in zip(signal_variances, background_variances):
    SNR = signal / background
    SNRs.append(SNR)

########################################################################

# Creating output csv file
# header for csv
header_list = ["Box Size", "Signal Variance", "Background Variance", "SNR"]

# zipping list, writing to data frame, writing to csv
combined_variance_list = zip(box_size_names, signal_variances, background_variances, SNRs) 
df = pd.DataFrame(combined_variance_list, columns = ['box_size_names', 'total_variances', 'masked_variances', 'SNRs'])
df.to_csv('variance_stat_mask_264.csv', index = False, header = header_list)