# 開発1：社内にたまっている文書の検索
このプログラムは、DOCXとXLSXファイルをUPLOADすると、類似した過去の資料を検索し、類似度順にリストアップします。Streamlitを使ってWEBアプリとしてサーブします。

# 環境構築
## 動作確認済み環境
- windows 11 home
## prerequisite/前提条件
- python 3.7.9 or higher
- pip
- pipenv
## 構築手順
- pipenv install docx2txt openpyxl sentence_transformers==1.1.1 protobuf==3.20 pandas streamlit

# 実行手順
1. dataフォルダを作成し、そこにdocxおよびxlsxファイルを置きます。フォルダ階層があってもOKです。
2. pipenv run python parse.py
    - parsed.csvが作成されます。
3. pipenv run python vector.py
    - feature.csvが作成されます
4. pipenv run python cosine.py
    - topN.csvが作成されます。
5. pipenv run streamlit run cosine_st.py
    - webアプリが起動します。http://localhost:8501 にアクセスします。

# 使い方
webブラウザでhttp://localhost:8501 にアクセスします。画面の指示に従いDOCXやXLSXファイルをアップロードします。アップロードしたファイルのテキストを解析し、dataフォルダにあるDOCXやXLSXファイルと照らし合わせて、内容が類似したものを類似度順に3点表示します。
