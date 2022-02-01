import numpy as np
from mdorado.vectors import norm_vecarray

vectorarray = np.array([[11,4,-4], [4,1,8], [-6,-7, 2], [3,0, -1]])
unitvectors, lengths = norm_vecarray(vecarray=vectorarray)
print(unitvectors)
print(lengths)
