import pandas as pd

df = pd.read_csv('ukb_filtered_hariri_mh_upto69.csv', sep=' ')
df_fc = pd.read_csv('ukb675871.bulk', sep=' ')

c1_values = list(set(df_fc.iloc[:, 0]))

df_fc_filtered = df[df['eid'].isin(c1_values)]

print(df_fc_filtered.columns)