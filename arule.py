import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from scipy.sparse import coo_matrix, csc_matrix
import scipy.sparse.sputils
import numpy as np
import Helper as hp

# df = pd.read_excel('Online Retail.xlsx')
# df.to_csv("OnlineRetail.csv", sep="\t", index=False)
# df = pd.read_csv("OnlineRetail.csv", sep="\t")
# df.to_csv("Transaktionen.csv", sep="\t", index=False)
df = pd.read_csv("Transaktionen.csv", sep="\t")

# query = """Select Year, OrderNo, iditemOrigin, Sales = sum(Sales)
#         from crhbusadwh01.infopool.fact.sales
#         group by Year, OrderNo, iditemOrigin"""
#
# con = hp.create_connection_string_turbo("crhbusadwh02", "AnalystCM")
# df = hp.sql_to_pandas(con, query)


df['iditemOrigin'] = df['iditemOrigin'].astype(str)
df['OrderNo'] = df['OrderNo'].astype(str)
df['Sales'] = df['Sales'].astype(float)
df['Year'] = df['Year'].astype(int)

check = (   (df['Year'] == 2017) &
            (df['Sales'] > 0) &
            (df['OrderNo'] > "") &
            (df['iditemOrigin']> "")
                )

df = df[check]

df["OrderNo"] = df['OrderNo'].astype("category")
df["iditemOrigin"] = df['iditemOrigin'].astype("category")
df['Sales'] = df['Sales'].astype('float')


basket = (df.groupby(['OrderNo', 'iditemOrigin'])['Sales']
          .sum()
          # .unstack()
          # .reset_index()
          # .fillna(0)
          # .set_index('OrderNo')
          )

basket


# df['InvoiceNo'] = df['InvoiceNo'].astype("category")
# df['Description'] = df['Description'].astype("category")
#
row = df['OrderNo'].cat.codes
col = df['iditemOrigin'].cat.codes
data = df['Sales'].astype(np.float)
sparse_matrix = coo_matrix((data, (row, col)))


#
# sparse_matrix.shape

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
basket_sets.drop('POSTAGE', inplace=True, axis=1)

frequent_itemsets = apriori(pd.DataFrame(sparse_matrix.toarray()), min_support=0.07)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()
