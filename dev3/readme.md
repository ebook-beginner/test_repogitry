# 開発3：社内にたまっている写真の検索
このプログラムは、写真ファイルをUPLOADすると、類似した過去の資料を検索し、類似度順にリストアップします。Streamlitを使ってWEBアプリとしてサーブします。

# 環境構築
## 動作確認済み環境
- windows 11 home
## prerequisite/前提条件
- python 3.7.9 or higher
- pip
- pipenv
## 構築手順
- pipenv install pandas scikit-learn streamlit ftfy regex tqdm git+https://github.com/openai/CLIP.git

# 実行手順
1. dataフォルダを作成し、そこに写真ファイルを置きます。フォルダ階層があってもOKです。
2. pipenv run python parse_feature.py
  - feature.csvが作成されます
3. pipenv run python cosine.py
  - topN.csvが作成されます。
4. pipenv run streamlit run cosine_st.py
  - webアプリが起動します。http://localhost:8501 にアクセスします。

# 使い方
webブラウザでhttp://localhost:8501 にアクセスします。画面の指示に従い写真ファイルをアップロードします。アップロードしたファイルを解析し、dataフォルダにある写真ファイルと照らし合わせて、内容が類似したものを類似度順に3点表示します。
