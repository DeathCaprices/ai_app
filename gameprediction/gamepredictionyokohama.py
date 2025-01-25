import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ページのタイトル
st.markdown("### 横浜ネオフリーバーズ野球試合予測サイト")

opponent_name = st.text_input("1.相手のチーム名を入力してください。:", max_chars=30)
st.write(f"横浜ネオフリーバーズと{opponent_name}の試合の予測を行います。")

# 事前に保存されたデータを読み込む
try:
    df1 = pd.read_csv("path_to_data/横浜ネオフリーバーズ_打撃データ.csv")
    df2 = pd.read_csv(f"path_to_data/{opponent_name}_打撃データ.csv")
    df3 = pd.read_csv("path_to_data/横浜ネオフリーバーズ_投手データ.csv")
    df4 = pd.read_csv(f"path_to_data/{opponent_name}_投手データ.csv")

    # データを小数点第3位まで丸める
    df1 = df1.round(3)
    df2 = df2.round(3)
    df3 = df3.round(2)
    df4 = df4.round(2)

    # データ表示
    st.markdown("##### 横浜ネオフリーバーズ打撃データ")
    st.dataframe(df1)
    st.markdown(f"##### {opponent_name}打撃データ")
    st.dataframe(df2)
    st.markdown("##### 横浜ネオフリーバーズ投手データ")
    st.dataframe(df3)
    st.markdown(f"##### {opponent_name}投手データ")
    st.dataframe(df4)

    # 特徴量の作成
    yokohama_features = np.array([df1["打率"].mean(), df1["本塁打"].mean(), df1["打点"].mean(), df3["防御率"].mean(), df3["三振"].mean(), df3["自責点"].mean()])
    opponent_features = np.array([df2["打率"].mean(), df2["本塁打"].mean(), df2["打点"].mean(), df4["防御率"].mean(), df4["三振"].mean(), df4["自責点"].mean()])

    # 特徴量を組み合わせる
    X = np.vstack([yokohama_features, opponent_features])
    y = [1, 0]  # ラベル: 1 (横浜勝利), 0 (相手勝利)

    # モデルの訓練
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # 予測の実行
    new_game_features = np.vstack([yokohama_features, opponent_features])
    prediction = model.predict(new_game_features)

    # 結果表示
    if st.button(f"{opponent_name}との試合結果を予測する"):
        if prediction[0] == 1:
            st.success("横浜ネオフリーバーズの勝利を予測しました！")
        else:
            st.markdown(f"{opponent_name}の勝利を予測しました！")

except Exception as e:
    st.error(f"データの読み込みに失敗しました: {e}")
    st.text("予測を実行するには、事前にデータが必要です。")







