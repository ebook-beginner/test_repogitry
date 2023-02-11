from sentence_transformers import util, SentenceTransformer
import pandas as pd

# モデルの読み込み
model = SentenceTransformer('stsb-xlm-r-multilingual')

# 文章の読み込み
df = pd.read_csv("parsed.csv", encoding='utf_8_sig')

# 埋め込みを取得
embedding = model.encode(df["sentence"])

# CSV出力
df_output = pd.concat([df["filepath"], pd.DataFrame(embedding)], axis=1)
print(df_output.head())
df_output.to_csv("feature.csv")
