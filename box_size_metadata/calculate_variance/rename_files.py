# rename_files.py
# The purpose of this script is to change the extension of each file

# Imports
import os

# Listing the files of a folder
folder = os.getcwd()
print('Before rename')
files = os.listdir(folder)
print(files)

# rename each file one by one
for file_name in files:
    
    # construct full file path
    old_name = os.path.join(folder, file_name)

    # Changing the extension from .mrc to _1.0.mrc
    new_name = old_name.replace('.mrc', '_1.0.mrc')
    os.rename(old_name, new_name)

# print new names
print('After rename')
print(os.listdir(folder))