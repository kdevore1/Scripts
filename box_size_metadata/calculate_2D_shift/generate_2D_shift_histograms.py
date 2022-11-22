# generate_2D_shift_histograms.py
# The purpose of this script is to create histograms of 2D shifts from cryoSPARC metadata (csv files)

# imports
import numpy as n
import pandas as pd
import matplotlib.pyplot as plt

###################################################################################################

# csv files to analyze
file_list = ['P49_box_1.0_2D_shift.csv', 'P49_box_1.5_2D_shift.csv', 'P49_box_2.0_2D_shift.csv',
             'P51_box_1.0_2D_shift.csv', 'P51_box_1.5_2D_shift.csv', 'P51_box_2.0_2D_shift.csv',
             'P52_box_1.0_2D_shift.csv', 'P52_box_1.5_2D_shift.csv', 'P52_box_2.0_2D_shift.csv']

# image files to be created
image_list = ['P49_box_1.0_2D_shift.png', 'P49_box_1.5_2D_shift.png', 'P49_box_2.0_2D_shift.png',
              'P51_box_1.0_2D_shift.png', 'P51_box_1.5_2D_shift.png', 'P51_box_2.0_2D_shift.png',
              'P52_box_1.0_2D_shift.png', 'P52_box_1.5_2D_shift.png', 'P52_box_2.0_2D_shift.png']

# output list
calculated_shift_list = []

# summary statistics list
mean_list = []
std_list = []
var_list = []


# loops through each csv file in the list, calculates the sqrt(x^2+y^2) for each row of df and appends to a list, 
for file, image in zip(file_list, image_list):
    df = pd.read_csv(file)
    
    # calculates the sqrt(x^2+y^2) for each row of df and appends to a list
    for ind in df.index:
        a = np.sqrt(df.x[ind]**2 + df.y[ind]**2)
        # a = df.x[ind] + df.y[ind]
        calculated_shift_list.append(a)
        
    # plots histogram for each list    
    plt.hist(calculated_shift_list)
    plt.ylabel('Counts')
    plt.xlabel('Coordinates')
    plt.title('Shifts')
    plt.savefig(image, dpi = 1200)
    plt.clf()
    
    # calculate summary statistics (mean, sigma, and variance) and write to separate lists
    mean = n.mean(calculated_shift_list)
    mean_list.append(mean)
    std = n.std(calculated_shift_list)
    std_list.append(std)
    var = n.var(calculated_shift_list)
    var_list.append(var)
    
    
# zip summary statistics lists
combined_list = zip(mean_list, std_list, var_list)

# header for csv file with summary statistics
sum_stat_header = ['Mean', 'Standard Deviation', 'Variance']

# convert to data frame and save as csv
df2 = pd.DataFrame(combined_list, columns = ['mean_list', 'std_list', 'var_list'])
df2.to_csv('sum_stat.csv', index = False, header = sum_stat_header)
