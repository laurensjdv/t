import pandas as pd
import numpy as np
from mrmr import mrmr_classif
from torch.utils.data import Dataset, DataLoader, Subset

from dataloaders.dataloader import FCMatrixDataset
ds = 'data/csv/severe_rds_v2.csv'
# ds = 'data/csv/rds_under_8_severe_v2.csv'
data = FCMatrixDataset(ds, 'data/fetched/25751/raw', '25751', None)

labels = np.genfromtxt(ds)
eids = labels[:,0]
labels = labels[:,1]
y = pd.DataFrame(labels)
dataloader = DataLoader(data, batch_size=1, shuffle=False)

x = np.array([sample[0].reshape(-1).numpy() for sample in dataloader])

x = pd.DataFrame(x)

features, relevance, redudancy = mrmr_classif(x, y, 55, return_scores=True)

# print(features)

# print if 0 is in features

print(0 in features)
print(features)

# old = np.array([1140, 536, 223, 907, 1449, 499, 1293, 45, 135, 1440, 879, 1384, 1210, 1316, 122, 22, 492, 638, 765, 1027, 1464, 501, 1462, 395, 26, 1079, 70, 425, 1403, 1409, 1318, 886, 1459, 1448, 939, 1163, 547, 10, 413, 676, 131, 216, 942, 1136, 1386, 232, 1455, 1337, 814, 139, 392, 1376, 1382, 471, 656]
#         )
# get overlap features and old
# print(np.intersect1d(features, old))
# print(len(np.intersect1d(features, old)))
# print(sorted(relevance, reverse=True)[:56])

