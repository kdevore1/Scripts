# generate_histograms.py
# The purpose of this script is to create 4 histograms of 3D poses from cryoSPARC metadata (csv files)
#   1-3) histograms with only the x poses, y poses, and z poses plotted individually for a particular box size
#   4) histogram with x poses, y poses, and z poses plotted together for a particular box size

# imports
import numpy as n
import csv as csv
from matplotlib import pyplot as plt
import pandas as pd

#-------------------------------------------------------------------------------------------------------------------
# generate histograms with only the x poses, y poses, and z poses plotted individually for a particular box size

# csv file names (generated from calculate_3D_poses.py)
file_names = ['P52_box_1.0.csv', 'P52_box_1.5.csv', 'P52_box_2.0.csv']


# loop through file names
for i in range(len(file_names)):
       
    # importing dataframe 
    df = pd.read_csv(file_names[i], header = 0)
    
    # extracting columns
    x = df.x
    y = df.y
    z = df.z

    # column list
    columns = [x, y, z]

    # x axis labels for histograms
    x_axis_labels = ['X Poses', 'Y Poses', 'Z Poses']

    # lists that will be appended with mean and st. dev. values obtained from individual histograms
    mu_columns = []
    sigma_columns = []

    # loop through columns in file and make histogram of poses
    for j in range(len(columns)):

        # making histogram 
        hist = plt.hist(columns[j], bins = 30)
        plt.ylabel('Counts')
        plt.xlabel(x_axis_labels[j])

        # lists with image names
        image_name_1 = ['P52_box_1.0_poses_x.png', 'P52_box_1.0_poses_y.png', 'P52_box_1.0_poses_z.png']
        image_name_2 = ['P52_box_1.5_poses_x.png', 'P52_box_1.5_poses_y.png', 'P52_box_1.5_poses_z.png']
        image_name_3 = ['P52_box_2.0_poses_x.png', 'P52_box_2.0_poses_y.png', 'P52_box_2.0_poses_z.png']

        # list that contains the image name lists 
        image_names = [image_name_1, image_name_2, image_name_3]

        # save figure of histogram 
        plt.savefig(image_names[i][j])
        plt.clf()

        # calculating mu and sigma of each
        mu_column = n.mean(columns[j])
        sigma_column = n.std(columns[j])
        mu_columns.append(mu_column)
        sigma_columns.append(sigma_column)
        
    # writing summary stats (mean and sd) for each box size to csv
    combined_list = zip(mu_columns, sigma_columns) 
    csv_file_names = ['P52_box_1.0_sum_stat.csv', 'P52_box_1.5_sum_stat.csv', 'P52_box_2.0_sum_stat.csv']   
    df = pd.DataFrame(combined_list, columns = ['mu_columns', 'sigma_columns'])
    df.to_csv(csv_file_names[i], index = False, header = False)

#-------------------------------------------------------------------------------------------------------------------
# generate histogram with x poses, y poses, and z poses plotted together for a particular box size

# image names for histograms with x, y, and z 
image_name_all = ['P52_box_1.0_poses_all.png', 'P52_box_1.5_poses_all.png', 'P52_box_2.0_poses_all.png']

# lists that will be appended with mean and st. dev. values obtained from individual histograms
mu_all = []
sigma_all = []

# plotting histogram of x, y, z (all)
for i, j in zip(file_names, image_name_all):

    # importing dataframe 
    df = pd.read_csv(i, header = 0)

    # extracting columns
    x = df.x
    y = df.y
    z = df.z

    # column list
    columns = [x, y, z]

    # making histogram with x, y, z
    hist_all = plt.hist(columns, bins=30, label=['X', 'Y', 'Z'])
    plt.ylabel('Counts')
    plt.xlabel('Poses')
    plt.legend(loc = 'upper right')

    # save figure of histogram 
    plt.savefig(j)
    plt.clf()
    
    # calculating mu and sigma of all
    mu_of_all = n.mean(columns)
    sigma_of_all = n.std(columns)
    mu_all.append(mu_of_all)
    sigma_all.append(sigma_of_all)

# combining mu_all and sigma_all lists in alternating fashion
combined_list_all = zip(mu_all, sigma_all)

# writing summary stats (mean and sd) for x, y, and z for each box size to csv
df = pd.DataFrame(combined_list_all, columns = ['mu_all', 'sigma_all'])
df.to_csv('P52_all_box_size_all_sum_stat.csv', index = False, header = False)