import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ページのタイトル
st.markdown("### 横浜ネオフリーバーズ野球試合予測サイト")

# 相手チーム名入力
opponent_name = st.text_input("1. 相手のチーム名を入力してください。:", max_chars=30)
st.write(f"横浜ネオフリーバーズと{opponent_name}の試合の予測を行います。")

# ファイルをバックエンドに組み込み
try:
    yokohama_batting = pd.read_csv("gameprediction/old_data/homebattingdata.csv").round(3)
    opponent_batting = pd.read_csv("gameprediction/old_data/opposingbattingdata.csv").round(3)
    yokohama_pitching = pd.read_csv("gameprediction/old_data/homepitchingdata.csv").round(2)
    opponent_pitching = pd.read_csv("gameprediction/old_data/opposingpitchingdata.csv").round(2)
except FileNotFoundError as e:
    st.error("必要なデータファイルが見つかりません。ファイルを確認してください。")
    st.stop()

# データの表示
st.markdown("##### 横浜ネオフリーバーズ打撃データ")
st.dataframe(yokohama_batting)

st.markdown(f"##### {opponent_name}打撃データ")
st.dataframe(opponent_batting)

st.markdown("##### 横浜ネオフリーバーズ投手データ")
st.dataframe(yokohama_pitching)

st.markdown(f"##### {opponent_name}投手データ")
st.dataframe(opponent_pitching)

# 特徴量の作成
yokohama_features = np.array([
    yokohama_batting["打率"].mean(),
    yokohama_batting["本塁打"].mean(),
    yokohama_batting["打点"].mean(),
    yokohama_pitching["防御率"].mean(),
    yokohama_pitching["三振"].mean(),
    yokohama_pitching["自責点"].mean(),
])

opponent_features = np.array([
    opponent_batting["打率"].mean(),
    opponent_batting["本塁打"].mean(),
    opponent_batting["打点"].mean(),
    opponent_pitching["防御率"].mean(),
    opponent_pitching["三振"].mean(),
    opponent_pitching["自責点"].mean(),
])

# 仮の訓練データ
X = np.vstack([yokohama_features, opponent_features])
y = [1, 0]  # 横浜勝利:1, 相手勝利:0

# モデルの訓練
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 予測実施
st.text(f"{opponent_name}との試合結果を予測する場合は下のボタンを押してください。")
if st.button(f"{opponent_name}との試合結果を予測する"):
    prediction = model.predict([yokohama_features, opponent_features])
    if prediction[0] == 1:
        st.success("横浜ネオフリーバーズの勝利を予測しました！")
    else:
        st.error(f"{opponent_name}の勝利を予測しました！\n横浜ネオフリーバーズは、戦略を再検討してください。")
