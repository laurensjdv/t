with open('gal_eids/gal_data.csv', 'r') as file:
    train = file.read().splitlines()

from tqdm.auto import tqdm

import pandas as pd

# with open('ukb675871.csv', 'r') as file:
#     # train = file.read().splitlines()
#     eids = []
#     i = 0
#     for line in tqdm(file):
#         eid = line[:10].split(',')[0]
#         # convert eid to int
#         eids.append(eid)



with open('rds_over_8.csv', 'r') as file:
    next(file)
    bulk = file.read().splitlines()




bulk_eid = [int(i.split(' ')[0]) for i in bulk]
bulk_label = [' '.join(i.split(' ')[1:]) for i in bulk]

# remove all " characters from bulk label
# bulk_label = [i.strip('"') for i in bulk_label]



print(bulk_eid[:10])
print(len(bulk_eid))
print(len(bulk_label))
print(bulk_label[:10])
bulk_dict = dict(zip(bulk_eid, bulk_label))




counts = {'0': 0, '1': 0}
# with open('balanced_sex_classification.csv', 'w') as file:
#     for item in bulk_dict:
#             if counts[bulk_dict[item]] < 5544:
#                 file.write(str(item) + ' ' + bulk_dict[item] + '\n')
#                 counts[bulk_dict[item]] += 1

# with open('bal_matched_rds_over_8.csv', 'r') as file:
# for item in bulk_dict:
#     print(item)

