import numpy as np

# print numpy versions
print(np.__version__)
np.show_config()

# Create a null vector of size 10
print(np.zeros(10))

# Create a null vector of size 10 but the fifth value which is 1
Z = np.zeros(10)
Z[4] = 1
print(Z)

# Create a vector with values ranging from 10 to 49
Z = np.arange(10,50)
print(Z)

#Reverse a vector
Z = np.arange(50)
Z = Z[::-1]

# create a 3x3 matrix with values ranging from 0 to 8
Z = np.arange(9).reshape(3,3)
print(Z)

# create a 3x3 identity matrix 
Z = np.eye(3)
print(Z)

# Create a 3x3x3 array with random values
Z = np.randm.random((3,3,3))
print(Z)

# Create a 10x10 array with random values and find the
minimum and maximum values
Z = np.random.random((10,10))
Zmin, Zmax = Z.min(), Z.max()
print(Zmin, Zmax)

# Create a random vector of size 30 and find the mean value
Z = np.random.random(30)
m = Z.mean()
print(m)

# Create a 2d array with 1 on the border and 0 inside
Z = np.ones(10,10)
Z[1:-1, 1:-1] = 0

# Create a 5x5 matrix with values 1,2,3,4 just below the
diagonal
Z = np.diag(1+np.arange(4),k=-1)
print(Z)

# Create a 8x8 matrix and fill it with a checkerboard pattern
Z = np.zeros((8,8),dtype=int)
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(Z)

# Consider a (6,7,8) shape array, what is the index (x,y,z) of
the 100th element?
print(np.unravel_index(100,(6,7,8)))

# Create a checkerboard 8x8 matrix using the tile function
Z = np.tile( np.array([[0,1],[1,0]]), (4,4))
print(Z)

# Normalize a 5x5 random matrix
Z = np.random.random((5,5))
Zmax, Zmin = Z.max(), Z.min()
Z = (Z - Zmin)/(Zmax - Zmin)
print(Z)

# Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)
Z = np.dot(np.ones((5,3)), np.ones((3,2)))
print(Z)

# Given a 1D array, negate all elements which are between 3
and 8, in place.
Z = np.arange(11)
Z[(3 < Z) & (Z <= 8)] *= -1

#How to round away from zero a float array ?
Z = np.random.uniform(-10,+10,10)
print (np.trunc(Z + np.copysign(0.5, Z)))

# Extract the integer part of a random array using 5 different
methods
Z = np.random.uniform(0,10,10)
print (Z - Z%1)
print (np.floor(Z))
print (np.ceil(Z)-1)
print (Z.astype(int))

# Create a 5x5 matrix with row values ranging from 0 to 4
Z = np.zeros((5,5))
Z += np.arange(6)
print(Z)

# Create a vector of size 10 with values ranging from 0 to 1,
both excluded
Z = np.linspace(0,1,12,endpoint=True)[1:-1]
print(Z)

# Create a random vector of size 10 and sort it
Z = np.random.random(10)
