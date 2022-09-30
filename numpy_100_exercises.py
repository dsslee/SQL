import numpy as np

# Q1) print numpy versions.
print(np.__version__)

# Show configuration.
np.show_config()

# Create a null vector of size 10.
print(np.zeros(10))

# Create a null vector of size 10 but the fifth value which is 1.
Z = np.zeros(10)
Z[4] = 1
print(Z)

# Create a vector with values ranging from 10 to 49.
Z = np.arange(10,50)
print(Z)

# Reverse a vector.
Z = np.arange(50)
Z = Z[::-1]

# Create a 3x3 matrix with values ranging from 0 to 8.
Z = np.arange(9).reshape(3,3)
print(Z)

# Find indices of non-zero elements from [1,2,0,0,4,0].
nz = np.nonzero([1,2,0,0,4,0])
print(nZ)

#Q10) Create a 3x3 identity matrix.
Z = np.eye(3)
print(Z)

# Create a 3x3x3 array with random values.
Z = np.randm.random((3,3,3))
print(Z)

# Create a 10x10 array with random values and find the
minimum and maximum values.
Z = np.random.random((10,10))
Zmin, Zmax = Z.min(), Z.max()
print(Zmin, Zmax)

# Create a random vector of size 30 and find the mean value.
Z = np.random.random(30)
m = Z.mean()
print(m)

# Create a 2d array with 1 on the border and 0 inside.
Z = np.ones(10,10)
Z[1:-1, 1:-1] = 0

# Create a 5x5 matrix with values 1,2,3,4 just below the
diagonal.
Z = np.diag(1+np.arange(4),k=-1)
print(Z)

# Create a 8x8 matrix and fill it with a checkerboard pattern.
Z = np.zeros((8,8),dtype=int)
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(Z)

# Consider a (6,7,8) shape array, what is the index (x,y,z) of
the 100th element?
print(np.unravel_index(100,(6,7,8)))

# Create a checkerboard 8x8 matrix using the tile function.
Z = np.tile( np.array([[0,1],[1,0]]), (4,4))
print(Z)

#Q20) Normalize a 5x5 random matrix.
Z = np.random.random((5,5))
Zmax, Zmin = Z.max(), Z.min()
Z = (Z - Zmin)/(Zmax - Zmin)
print(Z)

# Multiply a 5x3 matrix by a 3x2 matrix (real matrix product).
Z = np.dot(np.ones((5,3)), np.ones((3,2)))
print(Z)

# Given a 1D array, negate all elements which are between 3
and 8, in place.
Z = np.arange(11)
Z[(3 < Z) & (Z <= 8)] *= -1

# How to round away from zero a float array ?
Z = np.random.uniform(-10,+10,10)
print (np.trunc(Z + np.copysign(0.5, Z)))

# Extract the integer part of a random array using 5 different
methods.
Z = np.random.uniform(0,10,10)
print (Z - Z%1)
print (np.floor(Z))
print (np.ceil(Z)-1)
print (Z.astype(int))

# Create a 5x5 matrix with row values ranging from 0 to 4.
Z = np.zeros((5,5))
Z += np.arange(6)
print(Z)

# Q31) Create a vector of size 10 with values ranging from 0 to 1, both excluded.
Z = np.linspace(0,1,12,endpoint=True)[1:-1]
print(Z)

# Create a random vector of size 10 and sort it.
Z = np.random.random(10)
Z.sort()
print(Z)

# How to sum a small array faster than np.sum?
Z = np.arange(10)
np.add.reduce(Z)

# Consider two random array A anb B, check if they are equal.
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)
equal = np.allclose(A,B)
print(equal)

# Make an array immutable.
Z = np.zeros(10)
Z.flags.writeable = False
Z[0] = 1

# Consider a random 10x2 matrix representing cartesian
coordinates, convert them to polar coordinates.
Z = np.random.random((10,2))
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print(R)
print(T)

# Create random vector of size 10 and replace the maximum
value by 0.
Z = np.random.random(10)
Z[Z.argmax()] = 0
print(Z)

# Create a structured array with x and y coordinates covering
the [0,1]x[0,1] area.
Z = np.zeros((10,10), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,10),
np.linspace(0,1,10))
print(Z)

# Given two arrays, X and Y, construct the Cauchy matrix C
(Cij = 1/(xi - yj)).
X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))

#Q40) Print the minimum and maximum representable value for each numpy scalar type.
for dtype in [np.int8, np.int32, np.int64]:
print(np.iinfo(dtype).min)
print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
print(np.finfo(dtype).min)
print(np.finfo(dtype).max)
print(np.finfo(dtype).eps)

# How to print all the values of an array?
np.set_printoptions(threshold=np.nan)
Z = np.zeros((25,25))
print(Z)

# How to find the closest value (to a given scalar) in an array?
Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(Z[index])

# Create a structured array representing a position (x,y) and a
color (r,g,b).
Z = np.zeros(10, [ ('position', [ ('x', float, 1),
('y', float, 1)]),
('color', [ ('r', float, 1),
('g', float, 1),
('b', float, 1)])])
print(Z)

# How to convert a float (32 bits) array into an integer (32 bits) in place?
Z = np.arange(10, dtype=np.int32)
Z = Z.astype(np.float32, copy=False)

# How to read the following file?
# -------------
1,2,3,4,5
6,,,7,8
,,9,10,11
# -------------
Z = np.genfromtxt("missing.dat", delimiter=",")

# What is the equivalent of enumerate for numpy arrays?
Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
print(index, value)
for index in np.ndindex(Z.shape):
print(index, Z[index])

# Generate a generic 2D Gaussian-like array.
X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-((D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print(G)

# How to randomly place p elements in a 2D array?
n = 10
p = 3
Z = np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False),1)

# Q50) Subtract the mean of each row of a matrix.
X = np.random.rand(5, 10)
# Recent versions of numpy
Y = X - X.mean(axis=1, keepdims=True)
# Older versions of numpy
Y = X - X.mean(axis=1).reshape(-1, 1)

# How to I sort an array by the nth column?
Z = np.random.randint(0,10,(3,3))
print(Z[Z[:,1].argsort()])

# How to tell if a given 2D array has null columns?
Z = np.random.randint(0,3,(3,10))
print((~Z.any(axis=0)).any())

# Find the nearest value from a given value in an array.
Z = np.random.uniform(0,1,10)
z = 0.5
m = Z.flat[np.abs(Z - z).argmin()]
print(m)

# Create an array class that has a name attribute.
class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"): 
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")

Z = NamedArray(np.arange(10), "range_10")
print (Z.name)


# Consider a given vector, how to add 1 to each element
indexed by a second vector (be careful with repeated
indices)?
Z = np.ones(10)
I = np.random.randint(0,len(Z),20)
Z += np.bincount(I, minlength=len(Z))
print(Z)

# How to accumulate elements of a vector (X) to an array (F) based on an index list (I)?
X = [1,2,3,4,5,6]
I = [1,3,9,3,4,1]
F = np.bincount(I,X)
print(F)

# Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors.
w,h = 16,16
I = np.random.randint(0,2,(h,w,3)).astype(np.ubyte)
F = I[...,0]*256*256 + I[...,1]*256 +I[...,2]
n = len(np.unique(F))
print(np.unique(I))

# Considering a four dimensions array, how to get sum over the last two axis at once?
A = np.random.randint(0,10,(3,4,3,4))
sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
print(sum)

# Considering a one-dimensional vector D, how to compute
means of subsets of D using a vector S of same size
describing subset indices?
D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)

#Q60) How to get the diagonal of a dot product?
# option1: slowest
np.diag(np.dot(A, B))
# option2: faster
np.sum(A * B.T, axis=1)
# option3: fastest
np.einsum("ij,ji->i", A, B)

# Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value?
Z = np.array([1,2,3,4,5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[::nz+1] = Z
print(Z0)

# Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)?
A = np.ones((5,5,3))
B = 2*np.ones((5,5))
print(A * B[:,:,None])

# How to swap two rows of an array?
A = np.arange(25).reshape(5,5)
A[[0,1]] = A[[1,0]]
print(A)

# Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the triangles.
faces = np.random.randint(0,100,(10,3))
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print(G)

# Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C?
C = np.bincount([1,1,2,3,4,4,6])
A = np.repeat(np.arange(len(C)), C)
print(A)

# How to compute averages using a sliding window over an array?
def moving_average(a, n=3) :
ret = np.cumsum(a, dtype=float)
ret[n:] = ret[n:] - ret[:-n]
return ret[n - 1:] / n
Z = np.arange(20)
print(moving_average(Z, n=3))

# Consider a one-dimensional array Z, build a twodimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1.
from numpy.lib import stride_tricks
def rolling(a, window):
shape = (a.size - window + 1, window)
strides = (a.itemsize, a.itemsize)
return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)

# How to negate a boolean, or to change the sign of a float inplace?
Z = np.random.randint(0,2,100)
np.logical_not(arr, out=arr)
Z = np.random.uniform(-1.0,1.0,100)
np.negative(arr, out=arr)

# Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])?
def distance(P0, P1, p):
T = P1 - P0
L = (T**2).sum(axis=1)
U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
U = U.reshape(len(U),1)
D = P0 + U*T - p
return np.sqrt((D**2).sum(axis=1))
P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10,10,(1,2))
print(distance(P0, P1, p))

#Q70) Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])?
P0 = np.random.uniform(-10, 10, (10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10, 10, (10,2))
print np.array([distance(P0,P1,p_i) for p_i in p])

# Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]?
Z = np.arange(1,15,dtype=uint32)
R = stride_tricks.as_strided(Z,(11,4),(4,4))
print(R)

# Compute a matrix rank
Z = np.random.uniform(0,1,(10,10))
U, S, V = np.linalg.svd(Z) # Singular Value Decomposition
rank = np.sum(S > 1e-10)

# How to find the most frequent value in an array?
Z = np.random.randint(0,10,50)
print(np.bincount(Z).argmax())

# Extract all the contiguous 3x3 blocks from a random 10x10 matrix.
Z = np.random.randint(0,5,(10,10))
n = 3
i = 1 + (Z.shape[0]-3)
j = 1 + (Z.shape[1]-3)
C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
print(C)

# Create a 2D array subclass such that Z[i,j] == Z[j,i]
def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

s = symetric(np.random.randint(0,10,(5,5)))
s[2,3] = 42
print(S)
