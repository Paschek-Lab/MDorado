import numpy as np
from mdorado.vectors import vectormatrix

vectorarray = np.array([[11,4,-4], [4,1,8], [-6,-7, 2], [3,0, -1]])
vecmat = vectormatrix(apos=vectorarray, bpos=vectorarray)
print(vecmat)
