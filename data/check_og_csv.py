"""
Verifies that sex classification.csv is valid by loading ukb675871.csv
and comparing overlap between column 23 (31-0.0) to the sex_classification csv
"""


# load ukb675871.csv line for line and save entries 1 and 23 in a variable, then compare overlap with sex_classifcation.csv



import csv
import re
# load ukb675871.csv line for line and print entries from column 1 and 23 from first 100 rows

import csv





with open('ukb675871.csv', 'r') as file:
    reader = csv.reader(file)
    data = {}
    i = 0
    l =  next(reader, None)
    print(l[22])
    for row in reader:
        data[row[0]] = row[22]
        i+=1
        if i == 10000:
            break


with open('sex_classification.csv', 'r') as file:
    reader = csv.reader(file, delimiter=' ')
    next(reader, None)
    data2 = {}
    for row in reader:
        data2[row[0]] = row[1]

# data 1 and 2 overlap if the eid is key is the same and the value is the same
        
overlap = 0
max = 0
for key in data:
    if key in data2:
        max += 1
        if data[key] == data2[key]:
            overlap += 1

print(max)
print(overlap)
print(overlap/max)