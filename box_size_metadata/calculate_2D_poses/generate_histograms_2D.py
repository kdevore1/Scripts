# generate_histograms_2D.py
# The purpose of this script is to create histograms of 2D poses from cryoSPARC metadata (csv files)
# Summary statistics describing the poses are then calculated for each histogram

# imports
import numpy as n
from numpy import genfromtxt
import csv as csv
from matplotlib import pyplot as plt
import pandas as pd
#-------------------------------------------------------------------------------------------------------------------

# file name list
file_names = ['P49_box_1.0_2D.csv', 'P49_box_1.5_2D.csv', 'P49_box_2.0_2D.csv', 
'P51_box_1.0_2D.csv', 'P51_box_1.5_2D.csv', 'P51_box_2.0_2D.csv', 
'P52_box_1.0_2D.csv', 'P52_box_1.5_2D.csv', 'P52_box_2.0_2D.csv']

# image name
image_name_list = ['P49_box_1.0_2D_poses.png', 'P49_box_1.5_2D_poses.png', 'P49_box_2.0_2D_poses.png', 
'P51_box_1.0_2D_poses.png', 'P51_box_1.5_2D_poses.png', 'P51_box_2.0_2D_poses.png', 
'P52_box_1.0_2D_poses.png', 'P52_box_1.5_2D_poses.png', 'P52_box_2.0_2D_poses.png']

# summary statistics lists
mean_list = []
std_list = []
var_list = []


# plotting histogram, saving image, calculating summary statistics
for file, image_name in zip(file_names, image_name_list):
    
    # reads in data as numpy array
    data = genfromtxt(file, delimiter=',')
   
    plt.hist(data, bins=100)
    plt.ylabel('Counts')
    plt.xlabel('Poses')

    # save figure of histogram 
    plt.savefig(image_name)
    plt.clf()

    # calculate summary statistics (mean, sigma, and variance) and write to separate lists
    mean = n.mean(data)
    mean_list.append(mean)
    std = n.std(data)
    std_list.append(std)
    var = n.var(data)
    var_list.append(var)

# zip summary statistics lists
combined_list = zip(mean_list, std_list, var_list)

# header for csv file with summary statistics
sum_stat_header = ['Mean', 'Standard Deviation', 'Variance']

# convert to data frame and save as csv
df = pd.DataFrame(combined_list, columns = ['mean_list', 'std_list', 'var_list'])
df.to_csv('sum_stat.csv', index = False, header = sum_stat_header)