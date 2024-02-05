import pandas as pd

df = pd.read_csv('ukb_filtered_hariri_mh_upto69.csv', sep=' ')
df_fc = pd.read_csv('ukb675871.bulk', sep=' ', names = ['eid', 'entry'])

print(df_fc.head(5))
c1_values = list(set(df_fc.iloc[:, 0]))
print(list(df_fc.iloc[:, 0])[:10])
df_fc_filtered = df[df['eid'].isin(c1_values)]

# print(df_fc_filtered.shape)
df_fc_filtered = df_fc_filtered[df_fc_filtered['Lifetime dep Smith'] != '-1']
# print(df_fc_filtered.shape)

eids = list(df_fc_filtered['eid'])

df_fc = df_fc[df_fc['eid'].isin(eids)]
# print(df_fc_filtered[df_fc_filtered['Lifetime dep Smith'] == -1])
# print(df_fc_filtered['Lifetime dep stage'][:10])


# print(df_fc[df_fc['entry'] == '25750_2_0'])

df_fc.to_csv('ukb_filtered.bulk', sep=' ', index=False, header=False)