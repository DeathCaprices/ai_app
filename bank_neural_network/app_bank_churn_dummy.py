# app_bank_churn_dummy.py
import streamlit as st
import pandas as pd
import numpy as np

st.title("銀行解約予測アプリ（ダミーデータ版）")

# --------------------------
# ダミーデータの作成
# --------------------------
# ここでは顧客データを簡単なDataFrameで作成
data = pd.DataFrame({
    '年齢': [25, 40, 60, 35, 50],
    '残高': [1000, 5000, 3000, 4000, 2000],
    '口座開設年数': [1, 5, 10, 3, 7],
    '預金口座': [1, 0, 1, 1, 0],  # 1=あり, 0=なし
    '解約': [0, 1, 0, 1, 0]       # 0=継続, 1=解約
})

# --------------------------
# ユーザー入力用UI
# --------------------------
st.sidebar.header("顧客情報入力")
age = st.sidebar.number_input("年齢", min_value=18, max_value=100, value=30)
balance = st.sidebar.number_input("残高", min_value=0, max_value=10000, value=2000)
account_years = st.sidebar.number_input("口座開設年数", min_value=0, max_value=50, value=3)
has_deposit = st.sidebar.selectbox("預金口座の有無", options=[1, 0], format_func=lambda x: "あり" if x==1 else "なし")

# 入力データをDataFrame化
input_data = pd.DataFrame({
    '年齢': [age],
    '残高': [balance],
    '口座開設年数': [account_years],
    '預金口座': [has_deposit]
})

st.write("入力データ")
st.dataframe(input_data)

# --------------------------
# ダミー予測ロジック
# --------------------------
# 単純なルールで予測
# 例：残高が少ない or 口座開設年数が短い → 解約確率高
def dummy_predict(df):
    prob = []
    for _, row in df.iterrows():
        score = 0
        if row['残高'] < 3000:
            score += 0.5
        if row['口座開設年数'] < 5:
            score += 0.3
        if row['預金口座'] == 0:
            score += 0.2
        # 0.5以上で解約、未満で継続
        prob.append(1 if score >= 0.5 else 0)
    return prob

# --------------------------
# 予測ボタン
# --------------------------
if st.button("解約予測"):
    prediction = dummy_predict(input_data)
    result = "解約する可能性があります" if prediction[0] == 1 else "解約の可能性は低いです"
    st.success(result)
