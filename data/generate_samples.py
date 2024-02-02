import pandas as pd

df = pd.read_csv('ukb_filtered_hariri_mh_upto69.csv', sep=' ')
df_fc = pd.read_csv('ukb675871.bulk', sep=' ')

# print(df['eid'].head(20))
# print(df_fc.head(5))

print(df.shape)
print(df_fc.shape)

print(df_fc.head(20))
# for col in data.columns:
#     print(col)