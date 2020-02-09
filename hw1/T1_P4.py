#####################
# CS 181, Spring 2020
# Homework 1, Problem 4
# Start Code
##################

import csv
import numpy as np
import matplotlib.pyplot as plt
import math
from math import exp

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# TODO: basis functions
X1 = np.empty([years.size, 6])
for i in range(years.size):
    t = years[i]
    X1[i] = [1, t, t**2, t**3, t**4, t**5] 

X2 = np.empty([years.size, 12])
s = 1960
scale = 5
for i in range(years.size):
    t = years[i]
    X2[i] = [1,
             math.exp((-(years[i]-(s+(scale*0)))**2)/25),
             math.exp((-(years[i]-(s+(scale*1)))**2)/25),
             math.exp((-(years[i]-(s+(scale*2)))**2)/25),
             math.exp((-(years[i]-(s+(scale*3)))**2)/25),
             math.exp((-(years[i]-(s+(scale*4)))**2)/25),
             math.exp((-(years[i]-(s+(scale*5)))**2)/25),
             math.exp((-(years[i]-(s+(scale*6)))**2)/25),
             math.exp((-(years[i]-(s+(scale*7)))**2)/25),
             math.exp((-(years[i]-(s+(scale*8)))**2)/25),
             math.exp((-(years[i]-(s+(scale*9)))**2)/25),
             math.exp((-(years[i]-(s+(scale*10)))**2)/25),]

X3 = np.empty([years.size, 6])
for i in range(years.size):
    t = years[i]
    X3[i][0] = 1
    for j in range(1, 6, 1):
        X3[i][j] = math.cos(t/j)

X4 = np.empty([years.size, 26])
for i in range(years.size):
    t = years[i]
    X4[i][0] = 1
    for j in range(1, 26, 1):
        X4[i][j] = math.cos(t/j)

count = 0
for i in range(years.size):
    if years[i] > 1985:
        break
    count += 1
    
X12 = np.empty([count, 6])
for i in range(count):
    t = sunspot_counts[i]
    X12[i] = [1, t, t**2, t**3, t**4, t**5] 

X32 = np.empty([count, 6])
for i in range(count):
    t = sunspot_counts[i]
    X32[i][0] = 1
    for j in range(1, 6, 1):
        X32[i][j] = math.cos(t/j)
        
X42 = np.empty([count, 26])
for i in range(count):
    t = years[i]
    X42[i][0] = 1
    for j in range(1, 26, 1):
        X42[i][j] = math.cos(t/j)
        
# Nothing fancy for outputs.
Y = republican_counts
Y2 = sunspot_counts[years<last_year]

# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))
w1 = np.linalg.solve(np.dot(X1.T, X1) , np.dot(X1.T, Y))
w2 = np.linalg.solve(np.dot(X2.T, X2) , np.dot(X2.T, Y))
w3 = np.linalg.solve(np.dot(X3.T, X3) , np.dot(X3.T, Y))
w4 = np.linalg.solve(np.dot(X4.T, X4) , np.dot(X4.T, Y))
w12 = np.linalg.solve(np.dot(X12.T, X12) , np.dot(X12.T, Y2))
w32 = np.linalg.solve(np.dot(X32.T, X32) , np.dot(X32.T, Y2))
w42 = np.linalg.solve(np.dot(X42.T, X42) , np.dot(X42.T, Y2))

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)
grid_X = np.vstack((np.ones(grid_years.shape), grid_years))
grid_Yhat  = np.dot(grid_X.T, w)
grid_Yhat1 = np.dot(X1, w1.T)
grid_Yhat2 = np.dot(X2, w2.T)
grid_Yhat3 = np.dot(X3, w3.T)
grid_Yhat4 = np.dot(X4, w4.T)
grid_Yhat12 = np.dot(X12, w12.T)
grid_Yhat32 = np.dot(X32, w32.T)
grid_Yhat42 = np.dot(X42, w42.T)

# TODO: plot and report sum of squared error for each basis
def sse(yt, xt, wt):
    tsse = 0
    for i in range(len(yt)):
        tsse += (yt[i] - np.dot(wt.T, xt[i]))**2
    print(tsse)
# Plot the data and the regression line.
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.show()

plt.plot(years, republican_counts, 'o', years, grid_Yhat1, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.show()
sse(republican_counts, X1, w1)

plt.plot(years, republican_counts, 'o', years, grid_Yhat2, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.show()
sse(republican_counts, X2, w2)

plt.plot(years, republican_counts, 'o', years, grid_Yhat3, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.show()
sse(republican_counts, X3, w3)

plt.plot(years, republican_counts, 'o', years, grid_Yhat4, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.show()
sse(republican_counts, X4, w4)

plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o', sunspot_counts[years<last_year], grid_Yhat12, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()
sse(sunspot_counts[years<last_year], X12, w12)

plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o', sunspot_counts[years<last_year], grid_Yhat32, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()
sse(sunspot_counts[years<last_year], X32, w32)

plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o', sunspot_counts[years<last_year], grid_Yhat42, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()
sse(sunspot_counts[years<last_year], X42, w42)
