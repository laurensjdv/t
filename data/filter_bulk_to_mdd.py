import pandas as pd


# print(df_fc.head(5))
# c1_values = list(set(df_fc.iloc[:, 0]))
# print(list(df_fc.iloc[:, 0])[:10])
# df_fc_filtered = df[df['eid'].isin(c1_values)]

# print(df_fc_filtered.shape)
# df_fc_filtered = df_fc_filtered[df_fc_filtered['Lifetime dep Smith'] != '-1']
# print(df_fc_filtered.shape)

# eids = list(df_fc_filtered['eid'])

# df_fc = df_fc[df_fc['eid'].isin(eids)]
# print(df_fc_filtered[df_fc_filtered['Lifetime dep Smith'] == -1])
# print(df_fc_filtered['Lifetime dep stage'][:10])


# print(df_fc[df_fc['data-field'] == '25750_2_0'])

# df_fc.to_csv('ukb_filtered.bulk', sep=' ', index=False, header=False)

def generate_filtered_bulk(df, bulk_df):
    data_field = bulk_df['data-field'][0].split('_')[0]
    filename = 'ukb_filtered_' + data_field + '.bulk' 

    # get all ieds from bulk file
    datafield_ieds = list(set(bulk_df.iloc[:, 0]))
    # get subset of filtered dataset which overlaps with bulk ieds
    df_fc_filtered = df[df['eid'].isin(datafield_ieds)]
    # remove invalid samples
    df_fc_filtered = df_fc_filtered[df_fc_filtered['Lifetime dep Smith'] != '-1']
    # generate bulk dataframe
    filtered_eids = list(df_fc_filtered['eid'])
    bulk_df = bulk_df[bulk_df['eid'].isin(filtered_eids)]

    bulk_df.to_csv(filename, sep=' ', index=False, header=False)
    filename = 'ukb_filtered_' + data_field + '_harir_mh_upto69.csv'
    labels = df_fc_filtered[['eid', 'Lifetime dep Smith']]
    labels.to_csv(filename, sep = ' ', index=False)


df = pd.read_csv('ukb_filtered_hariri_mh_upto69.csv', sep=' ')
df_fc = pd.read_csv('ukb6771_25753.bulk', sep=' ', names = ['eid', 'data-field'])


generate_filtered_bulk(df, df_fc)
