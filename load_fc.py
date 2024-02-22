import numpy as np
import matplotlib.pyplot as plt

f = open('data/fetched/25750/1000032_25750_2_0.txt', "r")

fc = f.read()
fc_l = fc.split()
print(fc_l)
print(len(fc_l))
D = 21

sq_fc = np.zeros((D,D))
sq_fc[np.triu_indices(D,1)] = fc_l
sq_fc += sq_fc.T

print(sq_fc.shape)

plt.imshow(sq_fc)
plt.colorbar()
plt.show()