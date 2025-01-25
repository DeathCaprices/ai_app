import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from fractions import Fraction



# ページのタイトル
st.markdown("### 横浜ネオフリーバーズ野球試合予測サイト")

opponent_name= st.text_input("1.相手のチーム名を入力してください。:",max_chars=30)
st.write(f"横浜ネオフリーバーズと{opponent_name}の試合の予測を行います。")

st.text("2.左のサイドバーから①横浜ネオフリーバーズの打撃データをアップロードして下さい。")


# サイドバーにファイル選択のメッセージを表示
st.sidebar.markdown("### csvファイルの選択")
st.sidebar.write(f"横浜ネオフリーバーズと{opponent_name}のデータをアップロードしてください。")

# ユーザーがアップロードするファイル
uploaded_file1 = st.sidebar.file_uploader("①横浜ネオフリーバーズの打撃データ", type=["csv"])
uploaded_file2 = st.sidebar.file_uploader("②対戦相手の打撃データ", type=["csv"])
uploaded_file3 = st.sidebar.file_uploader("③横浜ネオフリーバーズの投手データ", type=["csv"])
uploaded_file4 = st.sidebar.file_uploader("④対戦相手の投手データ", type=["csv"])

# 1つ目のファイルがアップロードされたら処理
if uploaded_file1 is not None:
    df1 = pd.read_csv(uploaded_file1)
    df1 = df1.round(3)
    st.write("横浜ネオフリーバーズの打撃データ:", df1)

# 2つ目のファイルがアップロードされたら処理
if uploaded_file2 is not None:
    df2 = pd.read_csv(uploaded_file2)
    df2 = df2.round(3)
    st.write("対戦相手の打撃データ:", df2)

# 3つ目のファイルがアップロードされたら処理
if uploaded_file3 is not None:
    df3 = pd.read_csv(uploaded_file3)
    df3 = df3.round(3)
    st.write("横浜ネオフリーバーズの投手データ:", df3)

# 4つ目のファイルがアップロードされたら処理
if uploaded_file4 is not None:
    df4 = pd.read_csv(uploaded_file4)
    df4 = df4.round(3)
    st.write("対戦相手の投手データ:", df4)

# 必要な特徴量の計算
if uploaded_file1 is not None and uploaded_file2 is not None and uploaded_file3 is not None and uploaded_file4 is not None:
    # 特徴量計算
    new_yokohama_features = np.array([
        df1["打率"].mean(), df1["本塁打"].mean(), df1["打点"].mean(),
        df3["防御率"].mean(), df3["三振"].mean(), df3["自責点"].mean()
    ])
    new_opponent_features = np.array([
        df2["打率"].mean(), df2["本塁打"].mean(), df2["打点"].mean(),
        df4["防御率"].mean(), df4["三振"].mean(), df4["自責点"].mean()
    ])
    
    st.write("横浜ネオフリーバーズの特徴量:", old_yokohama_features)
    st.write("対戦相手の特徴量:", old_opponent_features)    # 特徴量の作成
    
    # 過去の試合データを仮の訓練データとして用意
    # 実際には事前に保存してある試合データをロード
    X = np.vstack([old_yokohama_features, old_opponent_features])  # 特徴量
    y = [1, 0]  # ラベル: 1 (横浜勝利), 0 (群馬勝利)
    
    # モデルの訓練
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X, y)
    


    # 最新の試合データを仮の訓練データとして用意
    df5 = pd.read_csv("newdata_batting_pitcher/new_homebattingdata.csv")
    df6 = pd.read_csv("newdata_batting_pitcher/new_opposingbattingdata.csv")    
    df7 = pd.read_csv("newdata_batting_pitcher/new_homepitchingdata.csv")
    df8 = pd.read_csv("newdata_batting_pitcher/new_opposingpitchingdata.csv")

    # 最新のデータの各チームの新データの打撃データと投手データから特徴量を抽出
    new_yokohama_features = np.array([df5["打率"].mean(), df5["本塁打"].mean(), df5["打点"].mean(), df7["防御率"].mean(), df7["三振"].mean(), df7["自責点"].mean()])
    new_opponent_features = np.array([df6["打率"].mean(), df6["本塁打"].mean(), df6["打点"].mean(), df8["防御率"].mean(), df8["三振"].mean(), df8["自責点"].mean()])

    new_game_features = np.vstack([new_yokohama_features, new_opponent_features])

    prediction = model.predict(new_game_features) 


    # 予測実施コメント
    st.text(f"{opponent_name}との試合結果を予測する場合は下のボタンを押して下さい。")

        
    # 結果を表示 st.markdownの場合は改行のため末尾(「{opponent_name}の勝利を予測しました！」の後に)にスペースを2つ入れてある
    if st.button(f"{opponent_name}との試合結果を予測する"):
        if prediction[0] == 1:
            st.success("横浜ネオフリーバーズの勝利を予測しました！")
        else:
            st.markdown(f"""{opponent_name}の勝利を予測しました！      
                    横浜ネオフリーバーズは、オーダーの組み換えや攻撃の方法、投手の起用法などを再検討して下さい。""")
