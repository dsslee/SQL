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