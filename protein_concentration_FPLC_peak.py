# Calculating Protein Concentration from FPLC Peak
# Intended use: Determine the concentration of a protein shown on an FPLC peak


# Variable Declaration
# A = area under the peak (ml * mAU)
A = float(input("What is the area under the peak?"))

# molar_epsi = extinction coefficient of your protein (Expasy ProtPram)
molar_epsi = float(input("What is the molar extinction coefficient?"))

# L = length 
L = 1 # g/L

# V = volume of the fractions collected (ml)
V = float(input("What is the volume of the fractions collected?"))

# MW = molecular weight of protein
MW = float(input("What is the molecular weight of the protein?"))


# Calculation of theoretical extinction coefficient 
theo_epsi = (molar_epsi * L) / MW


# Calculation of protein concentration (mg/ml)
C = A / (theo_epsi * L * V)
print("The concentration of your protein sample is ", C, "mg/ml")
