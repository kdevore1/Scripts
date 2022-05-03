# Rename (change file suffix)

# Intended use: change the suffix of the file

for f in *.undesirable_suffix
    do
    mv -- "$f" "${f%.undesirable_suffix}.new_suffix" 
done
