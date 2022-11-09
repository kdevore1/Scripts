# calculate_3D_poses.py
# The purpose of this script is to calculate 3D poses from cryoSPARC metadata and write into csv files for further manipulation

# imports
import numpy as n
from cryosparc_compute import dataset
import csv


# instatiate a new Dataset object -- comes from https://guide.cryosparc.com/setup-configuration-and-management/software-system-guides/manipulating-.cs-files-created-by-cryosparc
particle_dataset = dataset.Dataset()

# path names to iterate through -- change for each box size / project
path_names = ['/gludata_nfs2/kdevore/P49/exports/jobs/P49_J152_homo_refine_new/P49_J152_particles/P49_J152_particles_exported.cs',
              '/gludata_nfs2/kdevore/P49/exports/jobs/P49_J127_homo_refine_new/P49_J127_particles/P49_J127_particles_exported.cs',
              '/gludata_nfs2/kdevore/P49/exports/jobs/P49_J134_homo_refine_new/P49_J134_particles/P49_J134_particles_exported.cs']


# csv file names of output files -- change for each box size / project
file_names = ['P49_box_1.0.csv', 'P49_box_1.5.csv', 'P49_box_2.0.csv']

# header for csv file
pose_header = ['x', 'y', 'z']

# iterate through each path name and create a corresponding csv file 
for i, j in zip(path_names, file_names):

    # load the dataset into memory from file
    dataset_path = i

    # calling data
    particle_dataset.from_file(dataset_path)
    
    # taking 3D poses array
    poses = particle_dataset.from_file(dataset_path).data['alignments3D/pose']
    
    # writing to csv file
    with open(j, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(pose_header)
        writer.writerows(poses)