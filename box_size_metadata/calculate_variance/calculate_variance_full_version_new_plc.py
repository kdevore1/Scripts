# calculate_variance.py
# The purpose of this script is to read in a list of .mrc files calculate the background variances, signal variances, and SNRs. 
# These values will then be written to a .csv file.

# Version 0.0.2 - Po-Lin Chiu
#

########################################################################

# Imports
import numpy as np
import mrcfile
import csv
import glob
import time
import pandas as pd
import sys


def create_circular_mask(box_size, diam_pix):
    if box_size % 2 == 0:
        center = box / 2
        box = np.meshgrid(box_size, box_size)
        for i in xrange(box_size):
            for j in xrange(box_size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if dist > diam_pix:
                    box[i, j] = 0
                else:
                    box[i, j] = 1    
        return box
    
    else:
        print("The box should be in a size of even. ")
        return 0


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
    ## file = mrcfile.read(file)
    fmrc = mrcfile.open(file)
    data = fmrc.data
    h = fmrc.header

    # selects ONLY micrographs with particles picked
    ## if len(np.shape(data)) == 3: 
    if h.nz == 3:
        cmask = create_circular_mask(h.nx, diam)
        if cmask == 0: sys.exit("Box size needs to be even.")
        icmask = np.logical_not(cmask).astype(int)

        # create numpy arrays of zeros to store the intensity values
        ## background_data = np.zeros((np.shape(data)[1], np.shape(data)[2]))
        ## signal_data = np.zeros((np.shape(data)[1], np.shape(data)[2]))
        signal_data = np.multiply(data, cmask)
        background_data = np.multiply(data, icmask)

        # calculate the variance of the background intensity values
        background_variance = np.var(background_data)
        background_variances.append(background_variance)

        # calculate the variance of the signal intensity values
        signal_variance = np.var(signal_data)
        signal_variances.append(signal_variance)
    
    fmrc.close()
    

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