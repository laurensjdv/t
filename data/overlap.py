with open('gal_eids/gal_data.csv', 'r') as file:
    train = file.read().splitlines()

from tqdm.auto import tqdm

import re
import numpy as np

# with open('ukb675871.csv', 'r') as file:
#     # train = file.read().splitlines()
#     eids = []
#     i = 0
#     for line in tqdm(file):
#         eid = line[:10].split(',')[0]
#         # convert eid to int
#         eids.append(eid)



with open('ukb_filtered_hariri_mh_upto69.csv', 'r') as file:
    # skip first line
    l = file.readline()
    l = re.findall(r'[^"\s]+|"[^"]*"', l)
    # print entry in l with idx
    for idx, entry in enumerate(l):
        print(f"{idx}: {entry}")
    bulk = file.read().splitlines()

# remove first line from train
    
# train = eids[1:]
# train = [int(str(i.strip('"'))) for i in train]
# train.append(1193050)

# train = [int(i.split(' ')[0]) for i in train]

# bulk = [int(i.split(' ')[0]) for i in bulk]

# # get overlap between train and bulk

# overlap = list(set(train) & set(bulk))

# # print(set(train))
# print(len(overlap))
# print(len(train))

# print(bulk[:10])
# print(train[:10])

line = bulk[0]
print(line)
# split line by space except when it is in quotes
line = re.findall(r'[^"\s]+|"[^"]*"', line)
print(line)
print(f"eid: {line[0]}")
print(f"label: {line[1]}")
print(f"antidepressant count: {line[7]}")
print(f"SSRI count: {line[8]}")
print(f"T0 diagnosis: {line[20]}")
print(f"T1 diagnosis: {line[21]}")

def split_line(line):
    return re.findall(r'[^"\s]+|"[^"]*"', line)

antidepr_counts = np.sum([int(split_line(line)[7].strip('')) for line in bulk])
ssri_counts = np.sum([int(split_line(line)[8].strip('')) for line in bulk])

ssri_and_severe_counts = np.sum([int(split_line(line)[8].strip('')) for line in bulk if split_line(line)[21] == '"severe rMDD"'])
print(line[20])
print(f"Antidepressant counts: {antidepr_counts}")
print(f"SSRI counts: {ssri_counts}")
print(f"SSRI and severe counts: {ssri_and_severe_counts}")