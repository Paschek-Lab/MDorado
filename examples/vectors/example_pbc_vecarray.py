import numpy as np
from mdorado.vectors import pbc_vecarray

vectorarray = np.array([[11,4,-4], [4,1,8], [-6,-7, 2], [3,0, -1]])
box = [10,5,7]
pbc_vectorarray = pbc_vecarray(vecarray=vectorarray, box=box)
print(pbc_vectorarray)
