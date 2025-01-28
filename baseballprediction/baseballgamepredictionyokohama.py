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
st.write(f"○横浜ネオフリーバーズと{opponent_name}の試合の予測を行います。")

st.text("☆昨年の両チームの総合データを表示")


st.text("2.左のサイドバーから①横浜ネオフリーバーズの昨年の総合データをアップロードして下さい。")


# サイドバーにファイル選択のメッセージを表示
st.sidebar.markdown("### csvファイルの選択")
st.sidebar.write(f"横浜ネオフリーバーズと{opponent_name}のデータをアップロードしてください。")



# 1つ目のCSVファイルのアップロード

uploaded_file1 = st.sidebar.file_uploader("①横浜ネオフリーバーズの昨年の総合データ", type=["csv"])

if uploaded_file1 is not None:
    df1 = pd.read_csv(uploaded_file1)

    # 全ての小数点を第3位まで丸める
    df1 = df1.round(3)

    
    st.markdown("##### 横浜ネオフリーバーズ昨年の総合データ")
    st.dataframe(df1)



st.text(f"3.左のサイドバーから②{opponent_name}の昨年の総合データをアップロードして下さい。")


# 2つ目のCSVファイルのアップロード
uploaded_file2 = st.sidebar.file_uploader(f"②{opponent_name}の昨年の総合データ", type=["csv"])

if uploaded_file2 is not None:
    df2 = pd.read_csv(uploaded_file2)
    # 小数点をすべて第3位まで丸める
    df2 = df2.round(3)

    
    st.markdown(f"##### {opponent_name}昨年の総合データ")
    st.dataframe(df2)



st.text("☆直近の両チームの打撃・投手データを表示")


st.text("4.左のサイドバーから③横浜ネオフリーバーズの打撃データをアップロードして下さい。")


# 3つ目のCSVファイルのアップロード

uploaded_file3 = st.sidebar.file_uploader("③横浜ネオフリーバーズの打撃データ", type=["csv"])

if uploaded_file3 is not None:
    df3 = pd.read_csv(uploaded_file3)

    # 全ての小数点を第3位まで丸める
    df3 = df3.round(3)

    
    st.markdown("##### 横浜ネオフリーバーズ打撃データ")
    st.dataframe(df3)




st.text(f"5.左のサイドバーから④{opponent_name}の打撃データをアップロードして下さい。")


# 4つ目のCSVファイルのアップロード
uploaded_file4 = st.sidebar.file_uploader(f"④{opponent_name}の打撃データ", type=["csv"])

if uploaded_file4 is not None:
    df4 = pd.read_csv(uploaded_file4)
    # 小数点をすべて第3位まで丸める
    df4 = df4.round(3)

    
    st.markdown(f"##### {opponent_name}打撃データ")
    st.dataframe(df4)


st.text("6.左のサイドバーから⑤横浜ネオフリーバーズの投手データをアップロードして下さい。")

# 5つ目のCSVファイルのアップロード
uploaded_file5 = st.sidebar.file_uploader("⑤横浜ネオフリーバーズの投手データ", type=["csv"])

if uploaded_file5 is not None:
    df5 = pd.read_csv(uploaded_file5)
    
    # 小数点をすべて第2位まで丸める
    df5 = df5.round(2)

    st.markdown("##### 横浜ネオフリーバーズ投手データ")
    st.dataframe(df5)

st.text(f"7.左のサイドバーから⑥{opponent_name}の投手データをアップロードして下さい。")

# 6つ目のCSVファイルのアップロード
uploaded_file6 = st.sidebar.file_uploader(f"⑥{opponent_name}の投手データ", type=["csv"])

if uploaded_file6 is not None:
    df6 = pd.read_csv(uploaded_file6)
    # 小数点をすべて第2位まで丸める
    df6 = df6.round(2)  
       
    st.markdown(f"##### {opponent_name}投手データ")
    st.dataframe(df6)

    # 特徴量の作成
    # 例: 過去データから各チームの打撃データと投手データから特徴量を抽出
    old_yokohama_features = np.array([df1["打率"].mean(), df1["本塁打"].mean(), df1["打点"].mean(), df1["防御率"].mean(), df1["奪三振"].mean(), df1["自責点"].mean()])
    old_opponent_features = np.array([df2["打率"].mean(), df2["本塁打"].mean(), df2["打点"].mean(), df2["防御率"].mean(), df2["奪三振"].mean(), df2["自責点"].mean()])

    # 過去の試合データを仮の訓練データとして用意
    # 実際には事前に保存してある試合データをロード
    X = np.vstack([old_yokohama_features, old_opponent_features])  # 特徴量
    y = [1, 0]  # ラベル: 1 (横浜勝利), 0 (群馬勝利)
    
    # モデルの訓練
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X, y)
    


    
    # 最新のデータの各チームの新データの打撃データと投手データから特徴量を抽出
    new_yokohama_features = np.array([df3["打率"].mean(), df3["本塁打"].mean(), df3["打点"].mean(), df5["防御率"].mean(), df5["奪三振"].mean(), df5["自責点"].mean()])
    new_opponent_features = np.array([df4["打率"].mean(), df4["本塁打"].mean(), df4["打点"].mean(), df6["防御率"].mean(), df6["奪三振"].mean(), df6["自責点"].mean()])

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
