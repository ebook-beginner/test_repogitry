import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import streamlit as st
import base64
from io import BytesIO
import torch
import clip

# CLIPモデルの読み込み
model, preprocess = clip.load("ViT-B/32")
model.eval()

# CSVの読み込み
df_feature = pd.read_csv("feature.csv", encoding='utf_8_sig')

def convert_pil_base64(image, format="jpeg"):
    """pillow形式の画像をbase64文字列に変換する

    Args:
        image (Image): pillow形式の画像
        format (str, optional): 画像のエンコード. Defaults to "jpeg".

    Returns:
        str: base64文字列
    """
    buf = BytesIO()
    image.convert('RGB').save(buf, format)
    base64_string = base64.b64encode(buf.getvalue()).decode("ascii")

    return base64_string

def feature(pil_img):
    """Pillow形式の画像を受け取り、画像の埋め込みを計算して返す

    Args:
        pil_img (Image): Pillow形式の画像データ

    Returns:
        torch.tensor: 埋め込み
    """
    with torch.no_grad():
        image_input = torch.tensor(np.stack([preprocess(pil_img)]))
        image_features = model.encode_image(image_input).float()
    return image_features.reshape([1,-1])

def search(file):
    """検索を実行する

    Args:
        file : streamlitのupload画像
    """
    image = Image.open(file).convert('RGB')

    # 画像ファイルを開き、埋め込みに変換し、検索する。
    vec_feature = feature(image)
    
    # コサイン類似度を求める
    similarity = cosine_similarity(vec_feature, df_feature.iloc[:,2:])

    # いったん昇順にソートし、値とindexを取り出す。
    sorted_index = np.argsort(similarity)
    sorted_value = np.sort(similarity)

    # コサイン類似度の上位TopN個を取り出し、リストに格納する。
    topN = 3
    record = []
    columns=["query_index", "query_filepath", "rank", "similarity", "result_index", "result_filepath", "thumbnail"]
    for i, value in np.ndenumerate(sorted_index):
        rank = sorted_index.shape[1] - i[1] -1 #< 降順のランクを付ける。
        similarity = sorted_value[i[0], i[1]]
        result_index = value
        result_filepath = df_feature.iloc[result_index]["filepath"]
        
        if rank < topN:
            img_html = convert_pil_base64(Image.open(result_filepath))
            img_tag = f'<img src="data:image/jpeg;base64,{img_html}" width=100>'
            record.append([-1, file.name, rank, similarity, result_index, result_filepath, img_tag])

    # dataFrame化
    df_topN = pd.DataFrame(record, columns=columns)
    return df_topN


st.title("写真検索")
file = st.file_uploader("ファイルアップロード", type=['jpeg','jpg', 'png'])
if file is not None:
    df = search(file)
    if df is not None:
        # クエリ
        st.text("query")
        img_html = convert_pil_base64(Image.open(file))
        img_tag = f'<img src="data:image/jpeg;base64,{img_html}" width=100>'
        df_query = pd.DataFrame([[file.name, img_tag]])
        st.write(df_query.to_html(escape=False), unsafe_allow_html=True)

        # 検索結果
        st.text("result")
        df_result = df.sort_values("rank").loc[:,["rank", "similarity", "result_filepath", "thumbnail"]]
        st.write(df_result.to_html(escape=False), unsafe_allow_html=True)
    else:
        st.text("検索にヒットする画像は見つかりませんでした")