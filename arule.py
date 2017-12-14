import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from scipy.sparse import coo_matrix, csc_matrix
import scipy.sparse.sputils
import numpy as np

# df = pd.read_excel('Online Retail.xlsx')
# df.to_csv("OnlineRetail.csv", sep="\t", index=False)
df = pd.read_csv("OnlineRetail.csv", sep="\t")


df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]

df['Country'].unique()

df = df[df['Country'] =="Switzerland"]

basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum()
          .unstack()
          .reset_index()
          .fillna(0)
          .set_index('InvoiceNo'))

basket.shape

# df['InvoiceNo'] = df['InvoiceNo'].astype("category")
# df['Description'] = df['Description'].astype("category")
#
# row = df['InvoiceNo'].cat.codes
# col = df['Description'].cat.codes
# data = df['Quantity'].astype(np.int64)
# sparse_matrix = coo_matrix((data, (row, col)))
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
