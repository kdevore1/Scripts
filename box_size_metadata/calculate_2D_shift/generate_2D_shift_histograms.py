# generate_2D_shift_histograms.py
# The purpose of this script is to create histograms of 2D shifts from cryoSPARC metadata (csv files)

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# csv files to analyze
file_list = ['P49_box_1.0_2D_shift.csv', 'P49_box_1.5_2D_shift.csv', 'P49_box_2.0_2D_shift.csv',
             'P51_box_1.0_2D_shift.csv', 'P51_box_1.5_2D_shift.csv', 'P51_box_2.0_2D_shift.csv',
             'P52_box_1.0_2D_shift.csv', 'P52_box_1.5_2D_shift.csv', 'P52_box_2.0_2D_shift.csv']

# image files to be created
image_list = ['P49_box_1.0_2D_shift.png', 'P49_box_1.5_2D_shift.png', 'P49_box_2.0_2D_shift.png',
              'P51_box_1.0_2D_shift.png', 'P51_box_1.5_2D_shift.png', 'P51_box_2.0_2D_shift.png',
              'P52_box_1.0_2D_shift.png', 'P52_box_1.5_2D_shift.png', 'P52_box_2.0_2D_shift.png']


calculated_shift_list = []

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
