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

# Create a 3x3x3 array with random values
Z = np.randm.random((3,3,3))
print(Z)

