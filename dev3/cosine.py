import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# CSVの読み込み
df_feature = pd.read_csv("feature.csv", encoding='utf_8_sig')
# df_parsed = pd.read_csv("parsed.csv", encoding='utf_8_sig')

# コサイン類似度を求める
similarity = cosine_similarity(df_feature.iloc[:,2:2+768])

# 先頭5件をクエリ、後続をデータベースとみなす。
num_query = 5

# コサイン類似度のうち、クエリとデータベースに該当する箇所のみを取り出す。
trim = similarity[:num_query,num_query:]

# いったん昇順にソートし、値とindexを取り出す。
sorted_index = np.argsort(trim, axis=1)
sorted_value = np.sort(trim, axis=1)

# コサイン類似度の上位TopN個を取り出し、リストに格納する。
topN = 3
record = []
columns=["query_index", "query_filepath", "rank", "similarity", "result_index", "result_filepath"]
for i, value in np.ndenumerate(sorted_index):
    query_index = i[0]
    query_filepath = df_feature.iloc[query_index]["filepath"]
    rank = sorted_index.shape[1] - i[1] -1 #< 降順のランクを付ける。
    similarity = sorted_value[i[0], i[1]]
    result_index = value + num_query
    result_filepath = df_feature.iloc[result_index]["filepath"]
    
    if rank < topN:
        record.append([query_index, query_filepath, rank, similarity, result_index, result_filepath])

# dataFrame化し、CSVで保存
df_topN = pd.DataFrame(record, columns=columns)
df_topN.to_csv("topN.csv", encoding='utf_8_sig')


