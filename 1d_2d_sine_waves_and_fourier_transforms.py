# This script is meant to experiment with the generation of 1D and 2D sine waves + Fourier transforms

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt




# Plotting a 1D sine wave
# Creating a range of values 
x = np.arange(-500, 501, 1)

# Declaring variables
wavelength = 200

# 1D Sine wave
y = np.sin(2 * np.pi * x / wavelength)

# Plotting
plt.plot(x, y, "green")
plt.axes()
plt.show()




# Plotting a 2D sine wave
# Creating a range of values 
x = np.arange(-500, 501, 1)

# Creates a 2D representation of equations
X, Y = np.meshgrid(x, x)

# Declaring variables
wavelength = 200
angle = 0

# 2D Sine wave
# grating = np.sin(2 * np.pi * X / wavelength) ------ can use this equation if angle is not relevant
grating = np.sin(2*np.pi*(X*np.cos(angle) + Y*np.sin(angle)) / wavelength)

# Sets color map to grayscale
plt.set_cmap("gray")

# Plotting
plt.imshow(grating)
plt.show()




# Calculating a Fourier transform of a singular 2D sine wave
# Creating a range of values 
x = np.arange(-500, 501, 1)

# Creates a 2D representation of equations
X, Y = np.meshgrid(x, x)

# Declaring variables - useful for playing around with
wavelength = 200 # Smaller numbers cause distance between dots to shrink
angle = 0 

# 2D sine wave
grating = np.sin(2*np.pi*(X*np.cos(angle) + Y*np.sin(angle)) / wavelength)

# Calculate Fourier transform of grating
ft = np.fft.ifftshift(grating) # Centers the grating image
ft = np.fft.fft2(ft) # Fourier transform of function
ft = np.fft.fftshift(ft) # Centers the Fourier transform image

# Sets color map to grayscale
plt.set_cmap("gray")

# Plots 2D sine wave
plt.subplot(121)
plt.imshow(grating)

# Plots Fourier transform
plt.subplot(122)
plt.imshow(abs(ft))
# Note: Must zoom in to be able to see spots
plt.xlim([480, 520])
plt.ylim([520, 480])  # Order is reversed for y
plt.show()




# Adding two 2D sine waves together
# Creating a range of values 
x = np.arange(-500, 501, 1)

# Creates a 2D representation of equations
X, Y = np.meshgrid(x, x)

# Sine wave 1 parameters
wavelength_1 = 200
angle_1 = np.pi / 9
grating_1 = np.sin(2*np.pi*(X*np.cos(angle_1) + Y*np.sin(angle_1)) / wavelength_1)

# Sine wave 2 parameters
wavelength_2 = 100
angle_2 = 0
grating_2 = np.sin(2*np.pi*(X*np.cos(angle_2) + Y*np.sin(angle_2)) / wavelength_2)

# Plotting wave 1 and 2 
plt.set_cmap("gray")
plt.subplot(121)
plt.imshow(grating_1)
plt.subplot(122)
plt.imshow(grating_2)
plt.show()

# Adding waves together
gratings_combined = grating_1 + grating_2

# Calculating Fourier transform of combined waves
ft = np.fft.ifftshift(gratings_combined)
ft = np.fft.fft2(ft)
ft = np.fft.fftshift(ft)

# Creating figure with combined waves and Fourier transform of combined waves
plt.figure()
plt.subplot(121)
plt.imshow(gratings_combined)
plt.subplot(122)
plt.imshow(abs(ft))
plt.xlim([480, 520])
plt.ylim([520, 480])  # Order is reversed for y
plt.show()
