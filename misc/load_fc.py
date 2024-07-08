import numpy as np
import matplotlib.pyplot as plt

f = open('data/fetched/25751/raw/1000032_25751_2_0.txt', "r")

fc = f.read()
fc_l = fc.split()
print(fc_l)
print(len(fc_l))
D = 55

sq_fc = np.zeros((D,D))
sq_fc[np.triu_indices(D,1)] = fc_l
sq_fc += sq_fc.T

print(sq_fc.shape)

from math import sqrt

def compute_n1(c):
    return int((1 + sqrt(1 + 8 * c)) / 2)

# Example usage
n1 = compute_n1(1485)
print(n1)

plt.imshow(sq_fc)
plt.colorbar()
plt.show()