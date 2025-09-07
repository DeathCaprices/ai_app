import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Embedding, Flatten, Concatenate
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.title("銀行解約予測アプリ（Kaggleデータ体験版）")

# --- モデル選択 ---
model_type = st.sidebar.radio(
    "モデルを選択してください",
    ["数値データモデル", "埋め込み層ありモデル"]
)

st.sidebar.markdown("### 工夫点")
st.sidebar.write("""
- **Dropout**: 過学習を防ぐ  
- **BatchNormalization**: 学習を安定化  
- **Embedding**: カテゴリ変数を効率的に表現  
""")

# --- データ読み込み ---
@st.cache_data
def load_data():
    # 例: Kaggle Churn モデル用データ (csv を事前に同ディレクトリに置く)
    df = pd.read_csv("bank_churn/train.csv")
    return df

df = load_data()
st.subheader("サンプルデータ")
st.write(df.head())

# --- データ前処理 ---
X_num = df[["CreditScore", "Age", "Balance", "EstimatedSalary"]].values
X_cat = df[["Geography"]].values  # 例: 地域をカテゴリ変数に
y = df["Exited"].values

# カテゴリ変数を整数に変換
le = LabelEncoder()
X_cat = le.fit_transform(X_cat.reshape(-1)).reshape(-1, 1)

# --- 学習データ準備 ---
X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
    X_num, X_cat, y, test_size=0.2, random_state=42
)

# --- モデル定義 ---
def create_model():
    input_num = Input(shape=(X_num.shape[1],))
    x = Dense(10, activation="relu")(input_num)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(10, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(5, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    out = Dense(1, activation="sigmoid")(x)
    return Model(inputs=input_num, outputs=out)

def create_model_embedding():
    input_num = Input(shape=(X_num.shape[1],))
    x_num = Dense(10, activation="relu")(input_num)
    x_num = BatchNormalization()(x_num)
    x_num = Dropout(0.2)(x_num)

    input_cat = Input(shape=(1,))
    x_cat = Embedding(input_dim=len(le.classes_), output_dim=5)(input_cat)
    x_cat = Flatten()(x_cat)

    hidden = Concatenate()([x_num, x_cat])
    hidden = Dense(20, activation="relu")(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Dropout(0.1)(hidden)
    out = Dense(1, activation="sigmoid")(hidden)
    return Model(inputs=[input_num, input_cat], outputs=out)

# --- モデル選択と簡易学習 ---
if model_type == "数値データモデル":
    model = create_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_num_train, y_train, epochs=3, batch_size=32, verbose=0)
else:
    model = create_model_embedding()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit([X_num_train, X_cat_train], y_train, epochs=3, batch_size=32, verbose=0)

st.success("✅ サンプルデータでモデルを簡易学習しました")

# --- 入力方法選択 ---
input_mode = st.radio("入力方法を選択", ["サンプルから選ぶ", "手動入力"])

if input_mode == "サンプルから選ぶ":
    idx = st.number_input("サンプル行番号を選択 (0~10)", 0, 10, 0)
    st.write(df.iloc[idx])

    if model_type == "数値データモデル":
        sample_num = X_num[idx].reshape(1, -1)
        pred = model.predict(sample_num)
    else:
        sample_num = X_num[idx].reshape(1, -1)
        sample_cat = X_cat[idx].reshape(1, -1)
        pred = model.predict([sample_num, sample_cat])

else:
    if model_type == "数値データモデル":
        features = [st.number_input(col, float(df[col].mean())) for col in ["CreditScore", "Age", "Balance", "EstimatedSalary"]]
        pred = model.predict(np.array(features).reshape(1, -1))
    else:
        num_features = [st.number_input(col, float(df[col].mean())) for col in ["CreditScore", "Age", "Balance", "EstimatedSalary"]]
        cat_feature = st.selectbox("Geography", le.classes_)
        cat_idx = le.transform([cat_feature])[0]
        pred = model.predict([np.array(num_features).reshape(1, -1), np.array([cat_idx])])

# --- 結果表示 ---
st.subheader("予測結果")
if pred[0][0] > 0.5:
    st.error("➡ この顧客は **解約する可能性が高い** です。")
else:
    st.info("➡ この顧客は **解約しない可能性が高い** です。")
