#!/bin/bash

current_dir=`pwd`

# 1. Install the Nvidia graphic card driver
# sudo ubuntu-drivers devices
# sudo ubuntu-drivers autoinstall

# 2. Download CUDA and install the run file

# 3. Use apt to install required packages
sudo apt install build-essential cmake git libtiff-dev libtiff5-dev mpich curl

# 4. Compile FFTW library
cd ~/Downloads
wget https://www.fftw.org/fftw-3.3.10.tar.gz
tar zxvf fftw-3.3.10.tar.gz


# 5. Common packages that are used for structural biology
#   CCP4, Phenix, UCSF Chimera, UCSF ChimeraX, 
#
