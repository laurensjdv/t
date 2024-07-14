import numpy as np
import matplotlib.pyplot as plt

# f = open('data/fetched/25751/raw/1000032_25751_2_0.txt', "r")

fc = np.zeros([1485])
# mrmr_features = np.array([1140, 689, 427, 122, 139, 765, 907, 1384, 22, 1293, 1449, 492, 1440, 499, 1316, 1318, 135, 879, 886, 223, 1455, 676, 1136, 1464, 70, 1462, 1386, 939, 111, 413, 10, 1403, 1027, 547, 395, 1210, 942, 501, 45, 425, 638, 505, 26, 1409, 1448, 605, 232, 1459, 821, 1394, 1376, 809, 1163, 216, 791]
            # )
mrmr_features = np.array([1035, 382, 967, 1316, 34, 852, 761, 480, 875, 1357, 1203, 686, 415, 248, 1230, 1370, 1275, 466, 1274, 126, 840, 1470, 448, 629, 1292, 922, 617, 168, 946, 1372, 131, 219, 1247, 413, 72, 496, 880, 275, 863, 11, 571, 147, 877, 843, 92, 124, 1157, 1457, 249, 1126, 741, 1289, 1270, 1453, 742]
            )
fc[mrmr_features] = 1




# fc = f.read()
# fc_l = fc.split()
fc_l = fc
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
# plt.colorbar()
plt.show()