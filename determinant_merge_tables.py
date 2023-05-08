import pandas as pd

print('Loading Tables ...')
df_1_da = pd.read_excel('output/determinants/set1/da/set1_da_table.xlsx')
df_2_da = pd.read_excel('output/determinants/set2/da/set2_da_table.xlsx')
df_3_da = pd.read_excel('output/determinants/set3/da/set3_da_table.xlsx')

df_1_mape = pd.read_excel('output/determinants/set1/mape/set1_mape_table.xlsx')
df_2_mape = pd.read_excel('output/determinants/set2/mape/set2_mape_table.xlsx')
df_3_mape = pd.read_excel('output/determinants/set3/mape/set3_mape_table.xlsx')

df_1_dis = pd.read_excel('output/determinants/set1/dis/set1_dis_table.xlsx')
df_2_dis = pd.read_excel('output/determinants/set2/dis/set2_dis_table.xlsx')
df_3_dis = pd.read_excel('output/determinants/set3/dis/set3_dis_table.xlsx')

print('Exporting Tables ...')

with pd.ExcelWriter("output/determinants/lpc_determinants_tables.xlsx") as writer:
    df_1_da.to_excel(writer, sheet_name='SET1_DA', index=False)
    df_1_mape.to_excel(writer, sheet_name='SET1_MAPE', index=False)
    df_1_dis.to_excel(writer, sheet_name='SET1_DIS', index=False)

    df_2_da.to_excel(writer, sheet_name='SET2_DA', index=False)
    df_2_mape.to_excel(writer, sheet_name='SET2_MAPE', index=False)
    df_2_dis.to_excel(writer, sheet_name='SET2_DIS', index=False)

    df_3_da.to_excel(writer, sheet_name='SET3_DA', index=False)
    df_3_mape.to_excel(writer, sheet_name='SET3_MAPE', index=False)
    df_3_dis.to_excel(writer, sheet_name='SET3_DIS', index=False)

print('     ... Done!')
