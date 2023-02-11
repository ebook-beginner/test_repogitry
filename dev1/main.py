import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
from pathlib import Path

import docx2txt
import openpyxl
from sentence_transformers import util, SentenceTransformer
import streamlit as st

# CSVの読み込み
df_feature = pd.read_csv("feature.csv", encoding='utf_8_sig')
df_parsed = pd.read_csv("parsed.csv", encoding='utf_8_sig')

# モデルの読み込み
model = SentenceTransformer('stsb-xlm-r-multilingual')

def getEmbeddingByFile(file):
    """DOCXまたはXLSXファイルを読み込み、文章を取り出し、埋め込みを計算して返す。

    Args:
        file : DOCXまたはXLSXファイル
    """
    rows = []

    # DOCXファイル
    if file.name.lower().endswith(('.docx', '.DOCX')):
        # テンポラリファイルを作成して書き出し、そのファイルを読み込む。
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
            fp = Path(tmp_file.name)
            fp.write_bytes(file.getvalue())
            text = docx2txt.process(tmp_file.name)
        if text is None:
            return None
        elif len(text) == 1:
            rows.append([file, text])
        elif len(text) > 1:
            text2 = [t.replace("\n", "") for t in text]
            text2 = ''.join(text2)
            rows.append([file, text2])

    # XLSXファイル
    elif file.name.lower().endswith(('.xlsx', '.XLSX')):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
            fp = Path(tmp_file.name)
            fp.write_bytes(file.getvalue())
            wb = openpyxl.load_workbook(tmp_file.name, data_only=True)

        buffer = []
        for ws in wb.worksheets:
            for cells in tuple(ws.rows):
                for cell in cells:
                    if cell.value is not None:
                        buffer.append(str(cell.value).replace("\n", ""))
        buffer = ' '.join(buffer)
        rows.append([file, buffer])
        
    else:
        return None
    
    df = pd.DataFrame(rows, columns=["filepath", "sentence"])

    # 埋め込みを取得
    embedding = model.encode(df["sentence"])

    return df["sentence"], embedding

def search(file):
    """検索を実行する

    Args:
        file : DOCX or XLSXのファイル
    """
    # DOCX or XLSXのファイルを開き、文章を取得し、埋め込みに変換し、検索する。
    text, df_query = getEmbeddingByFile(file)
    
    # コサイン類似度を求める
    similarity = cosine_similarity(df_query, df_feature.iloc[:,2:2+768])

    # いったん昇順にソートし、値とindexを取り出す。
    sorted_index = np.argsort(similarity)
    sorted_value = np.sort(similarity)

    # コサイン類似度の上位TopN個を取り出し、リストに格納する。
    topN = 3
    record = []
    columns=["query_index", "query_filepath", "query_sentence", "rank", "similarity", "result_index", "result_filepath", "result_sentence"]
    for i, value in np.ndenumerate(sorted_index):
        query_index = i[0]
        query_filepath = df_parsed.iloc[query_index]["filepath"]
        query_sentence = df_parsed.iloc[query_index]["sentence"][:40]
        rank = sorted_index.shape[1] - i[1] -1 #< 降順のランクを付ける。
        similarity = sorted_value[i[0], i[1]]
        result_index = value
        result_filepath = df_parsed.iloc[result_index]["filepath"]
        result_sentence = df_parsed.iloc[result_index]["sentence"][:40]
        
        if rank < topN:
            record.append([-1, file.name, text, rank, similarity, result_index, result_filepath, result_sentence])

    # dataFrame化
    df_topN = pd.DataFrame(record, columns=columns)
    return df_topN


st.title("文書検索")
file = st.file_uploader("ファイルアップロード", type=['xlsx','docx'])
if file is not None:
    df = search(file)
    if df is not None:
        # クエリ
        st.text("query")
        st.dataframe(df.iloc[0].loc[["query_filepath", "query_sentence"]])
        # 検索結果
        st.text("result")
        st.dataframe(df.sort_values("rank").loc[:,["rank", "similarity", "result_filepath", "result_sentence"]])
    else:
        st.text("検索にヒットする文書は見つかりませんでした")