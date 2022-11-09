calculate_3D_poses.py should be ran on the workstation in order to generate csv files containing the pose (x, y, and z coordinate information) for each particle.
This script will need to be refactored and run for every project/dataset.

Once csv files are generated, generate_histograms.py should be ran to generate histograms of the pose information (in .png format). 
This script will also generate 2 types of .csv files:
	1) A "sum_stat" file for each box size containing miu and sigma values of the x, y, and z poses
		(each row will be either the x, y, or z)
	2) An "all_sum_stat" file for each box size containing a miu and sigma value for the x, y, and z pose distribution combined 
		(each row will be a different box size)
This script will need to be refactored and run for every project/dataset.
