import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from fractions import Fraction
import matplotlib.font_manager as fm
import os



# ページのタイトル
st.markdown("### 群馬ニューフリーバーズ野球試合予測サイト")

# タブの作成
tabs = st.tabs(["試合予測", "昨年のデータ", "直近のデータ","試合結果予測グラフ"])



with tabs[0]:  # 試合予測

    # 相手チーム名を入力
    opponent_name= st.text_input("〇相手のチーム名を入力してください。:",max_chars=30)

    # 入力がない場合にエラーを表示しつつ、デフォルト名を使用
    if not opponent_name:
        st.error("相手のチーム名を入力しないと相手のチーム名が表示されません。")
        opponent_name = "相手チーム"  # デフォルトの名前を設定

    st.write(f"○群馬ニューフリーバーズと{opponent_name}の試合の予測を行います。")
    # ※データは必ず①から⑥の順にアップロードして下さい。の次に半角スペースを2つ入れて改行。
    st.markdown("・試合予測方法  \n下の1から6の順に従い、左のサイドバーから両チームのデータのアップロードを行います。  \n※データは必ず①から⑥の順にアップロードして下さい。  \n順番を誤ると試合の予測が不可能となります。")
    
    st.text("☆昨年の両チームの総合データを表示")


    st.text("1.左のサイドバーから①群馬ニューフリーバーズの昨年の総合データをアップロードして下さい。")


    # サイドバーにファイル選択のメッセージを表示
    st.sidebar.markdown("### csvファイルの選択")
    st.sidebar.write(f"群馬ニューフリーバーズと{opponent_name}のデータをアップロードしてください。")



    # 1つ目のCSVファイルのアップロード

    uploaded_file1 = st.sidebar.file_uploader("①群馬ニューフリーバーズの昨年の総合データ", type=["csv"])

    if uploaded_file1 is not None:
        df1 = pd.read_csv(uploaded_file1)

        # 全ての小数点を第3位まで丸める
        df1 = df1.round(3)

        
        st.markdown("##### 群馬ニューフリーバーズ昨年の総合データ")
        st.dataframe(df1)



    st.text(f"2.左のサイドバーから②{opponent_name}の昨年の総合データをアップロードして下さい。")


    # 2つ目のCSVファイルのアップロード
    uploaded_file2 = st.sidebar.file_uploader(f"②{opponent_name}の昨年の総合データ", type=["csv"])

    if uploaded_file2 is not None:
        df2 = pd.read_csv(uploaded_file2)
        # 小数点をすべて第3位まで丸める
        df2 = df2.round(3)

        
        st.markdown(f"##### {opponent_name}昨年の総合データ")
        st.dataframe(df2)



    st.text("☆直近の両チームの打撃・投手データを表示")


    st.text("3.左のサイドバーから③群馬ニューフリーバーズの打撃データをアップロードして下さい。")


    # 3つ目のCSVファイルのアップロード

    uploaded_file3 = st.sidebar.file_uploader("③群馬ニューフリーバーズの打撃データ", type=["csv"])

    if uploaded_file3 is not None:
        df3 = pd.read_csv(uploaded_file3)

        # 全ての小数点を第3位まで丸める
        df3 = df3.round(3)

        
        st.markdown("##### 群馬ニューフリーバーズ打撃データ")
        st.dataframe(df3)




    st.text(f"4.左のサイドバーから④{opponent_name}の打撃データをアップロードして下さい。")


    # 4つ目のCSVファイルのアップロード
    uploaded_file4 = st.sidebar.file_uploader(f"④{opponent_name}の打撃データ", type=["csv"])

    if uploaded_file4 is not None:
        df4 = pd.read_csv(uploaded_file4)
        # 小数点をすべて第3位まで丸める
        df4 = df4.round(3)

        
        st.markdown(f"##### {opponent_name}打撃データ")
        st.dataframe(df4)


    st.text("5.左のサイドバーから⑤群馬ニューフリーバーズの投手データをアップロードして下さい。")

    # 5つ目のCSVファイルのアップロード
    uploaded_file5 = st.sidebar.file_uploader("⑤群馬ニューフリーバーズの投手データ", type=["csv"])

    if uploaded_file5 is not None:
        df5 = pd.read_csv(uploaded_file5)
        
        # 小数点をすべて第2位まで丸める
        df5 = df5.round(2)

        st.markdown("##### 群馬ニューフリーバーズ投手データ")
        st.dataframe(df5)

    st.text(f"6.左のサイドバーから⑥{opponent_name}の投手データをアップロードして下さい。")

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
        old_gumna_features = np.array([df1["打率"].mean(), df1["本塁打"].mean(), df1["打点"].mean(), df1["防御率"].mean(), df1["奪三振"].mean(), df1["自責点"].mean()])
        old_opponent_features = np.array([df2["打率"].mean(), df2["本塁打"].mean(), df2["打点"].mean(), df2["防御率"].mean(), df2["奪三振"].mean(), df2["自責点"].mean()])

        # 過去の試合データを仮の訓練データとして用意
        # 実際には事前に保存してある試合データをロード
        X = np.vstack([old_gumna_features, old_opponent_features])  # 特徴量
        y = [1, 0]  # ラベル: 1 (横浜勝利), 0 (群馬勝利)
        
        # モデルの訓練
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X, y)
        


        
        # 最新のデータの各チームの新データの打撃データと投手データから特徴量を抽出
        new_gumna_features = np.array([df3["打率"].mean(), df3["本塁打"].mean(), df3["打点"].mean(), df5["防御率"].mean(), df5["奪三振"].mean(), df5["自責点"].mean()])
        new_opponent_features = np.array([df4["打率"].mean(), df4["本塁打"].mean(), df4["打点"].mean(), df6["防御率"].mean(), df6["奪三振"].mean(), df6["自責点"].mean()])

        new_game_features = np.vstack([new_gumna_features, new_opponent_features])

        prediction = model.predict(new_game_features)
              

        # 予測実施コメント
        st.text(f"{opponent_name}との試合結果を予測する場合は下のボタンを押して下さい。  \n※試合結果予測グラフのページでは、データ①から⑥のアップロード後、  \n即時に試合結果の予測を見る事が出来ます。")

            
        # 結果を表示 st.markdownの場合は改行のため末尾(「{opponent_name}の勝利を予測しました！」の後に)にスペースを2つ入れてある
        if st.button(f"{opponent_name}との試合結果を予測する"):
            if prediction[0] == 1:
                st.success("群馬ニューフリーバーズの勝利を予測しました！")
            else:
                st.markdown(f"""{opponent_name}の勝利を予測しました！      
                        群馬ニューフリーバーズは、オーダーの組み換えや攻撃の方法、投手の起用法などを再検討して下さい。""")


# フォントファイルを指定

BASE_DIR = os.path.dirname(__file__)
font_path = os.path.join(BASE_DIR, "fonts", "NotoSansJP-Regular.ttf")
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()




# データ可視化タブの処理を限定
with tabs[1]:  
    st.markdown("### 両チームの昨年のデータ")

    if 'df1' in globals() and 'df2' in globals():
        # 2つの散布図を作成する (2行1列)
        fig, ax = plt.subplots(2, 1, figsize=(8, 12))  # 2行1列のグラフ
        

        # 各特徴量ごとにプロット（色を変える）
        ax[0].scatter(df1["安打"], df2["安打"], label="安打", color="blue")
        ax[0].scatter(df1["本塁打"], df2["本塁打"], label="本塁打", color="red")
        ax[0].scatter(df1["打点"], df2["打点"], label="打点", color="green")
        ax[0].scatter(df1["盗塁"], df2["盗塁"], label="盗塁", color="orange")
        ax[0].scatter(df1["犠打"], df2["犠打"], label="犠打", color="purple")
        ax[0].scatter(df1["四死球"], df2["四死球"], label="四死球", color="brown")
        ax[0].scatter(df1["三振"], df2["三振"], label="三振", color="pink")
        
        # 軸ラベルと凡例
        ax[0].set_title("両チームの昨年の打撃データ")
        ax[0].set_xlabel("群馬ニューフリーバーズ")
        ax[0].set_ylabel(f"{opponent_name}")
        ax[0].legend()
        
        

        # 各特徴量ごとにプロット（色を変える）
        ax[1].scatter(df1["被安打"], df2["被安打"], label="被安打", color="blue")
        ax[1].scatter(df1["被本塁打"], df2["被本塁打"], label="被本塁打", color="red")
        ax[1].scatter(df1["与四死球"], df2["与四死球"], label="与四死球", color="green")
        ax[1].scatter(df1["奪三振"], df2["奪三振"], label="奪三振", color="orange")
        ax[1].scatter(df1["失点"], df2["失点"], label="失点", color="purple")
        ax[1].scatter(df1["自責点"], df2["自責点"], label="自責点", color="brown")
        
        # 軸ラベルと凡例
        ax[1].set_title("両チームの昨年の投手データ")
        ax[1].set_xlabel("群馬ニューフリーバーズ")
        ax[1].set_ylabel(f"{opponent_name}")
        ax[1].legend()
        # Streamlit に表示
        st.pyplot(fig)

        # 防御率と打率の棒グラフを作成
        fig_bar, ax_bar = plt.subplots(figsize=(6, 6))

        # チーム名
        teams = ["群馬ニューフリーバーズ", opponent_name]
        # 打率と防御率のデータ
        batting_avg = [df1["打率"].mean(), df2["打率"].mean()]
        era = [df1["防御率"].mean(), df2["防御率"].mean()]

        # 棒グラフを描画
        x = np.arange(len(teams))  # X軸の位置
        width = 0.35  # バーの幅

        ax_bar.bar(x - width/2, batting_avg, width, label="打率", color="blue")
        ax_bar.bar(x + width/2, era, width, label="防御率", color="red")

        # 軸とタイトル
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(teams)
        ax_bar.set_ylabel("値")
        ax_bar.set_title("両チームの昨年の防御率と打率の比較")
        ax_bar.legend()

        
        # Streamlit に表示
        st.pyplot(fig_bar)
    
    else:
        st.warning("データをアップロードしていない場合、左のサイドバーから必ず①②の順に両チームの昨年の総合データをアップロードして下さい。")
        


       
with tabs[2]:  # 両チームの直近のデータ   
    st.markdown("### 両チームの直近のデータ")

    if 'df3' in globals() and 'df4' in globals() and 'df5' in globals() and 'df6' in globals():
        # 2つの散布図を作成する (2行1列)
        fig, ax = plt.subplots(2, 1, figsize=(8, 12))  # 2行1列のグラフ
        

        # 各特徴量ごとにプロット（色を変える）
        ax[0].scatter(df3["安打"].sum(), df4["安打"].sum(), label="安打", color="blue")
        ax[0].scatter(df3["本塁打"].sum(), df4["本塁打"].sum(), label="本塁打", color="red")
        ax[0].scatter(df3["打点"].sum(), df4["打点"].sum(), label="打点", color="green")
        ax[0].scatter(df3["盗塁"].sum(), df4["盗塁"].sum(), label="盗塁", color="orange")
        ax[0].scatter(df3["犠打"].sum(), df4["犠打"].sum(), label="犠打", color="purple")
        ax[0].scatter(df3["四死球"].sum(), df4["四死球"].sum(), label="四死球", color="brown")
        ax[0].scatter(df3["三振"].sum(), df4["三振"].sum(), label="三振", color="pink")
        
        # 軸ラベルと凡例
        ax[0].set_title("両チームの直近の打撃データ")
        ax[0].set_xlabel("群馬ニューフリーバーズ")
        ax[0].set_ylabel(f"{opponent_name}")
        ax[0].legend()
        
        

        # 各特徴量ごとにプロット（色を変える）
        ax[1].scatter(df5["被安打"].sum(), df6["被安打"].sum(), label="被安打", color="blue")
        ax[1].scatter(df5["被本塁打"].sum(), df6["被本塁打"].sum(), label="被本塁打", color="red")
        ax[1].scatter(df5["与四死球"].sum(), df6["与四死球"].sum(), label="与四死球", color="green")
        ax[1].scatter(df5["奪三振"].sum(), df6["奪三振"].sum(), label="奪三振", color="orange")
        ax[1].scatter(df5["失点"].sum(), df6["失点"].sum(), label="失点", color="purple")
        ax[1].scatter(df5["自責点"].sum(), df6["自責点"].sum(), label="自責点", color="brown")
        
        # 軸ラベルと凡例
        ax[1].set_title("両チームの直近の投手データ")
        ax[1].set_xlabel("群馬ニューフリーバーズ")
        ax[1].set_ylabel(f"{opponent_name}")
        ax[1].legend()
        # Streamlit に表示
        st.pyplot(fig)

        # 防御率と打率の棒グラフを作成
        fig_bar, ax_bar = plt.subplots(figsize=(6, 6))

        # チーム名
        teams = ["群馬ニューフリーバーズ", opponent_name]
        # 打率と防御率のデータ
        batting_avg = [df3["打率"].mean(), df4["打率"].mean()]
        era = [df5["防御率"].mean(), df6["防御率"].mean()]

        # 棒グラフを描画
        x = np.arange(len(teams))  # X軸の位置
        width = 0.35  # バーの幅

        ax_bar.bar(x - width/2, batting_avg, width, label="打率", color="blue")
        ax_bar.bar(x + width/2, era, width, label="防御率", color="red")

        # 軸とタイトル
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(teams)
        ax_bar.set_ylabel("値")
        ax_bar.set_title("両チームの直近の防御率と打率の比較")
        ax_bar.legend()

        
        # Streamlit に表示
        st.pyplot(fig_bar)
    
    else:
        st.warning("データをアップロードしていない場合、左のサイドバーから必ず①から⑥の順にアップロードして下さい。\n\n"
                   "また既にデータ①②をアップロードしている場合は、残りのデータを必ず③から⑥の順にアップロードして下さい。")
        


with tabs[3]:  # 試合結果グラフ
    st.markdown("### 試合結果の可視化")
    # 6つのデータフレームがすべて存在するかチェック
    if 'df1' in globals() and 'df2' in globals() and 'df3' in globals() and 'df4' in globals() and 'df5' in globals() and 'df6' in globals():
    # 勝率を取得（モデルが確率を出力できる場合）
        if hasattr(model, "predict_proba") and 'new_game_features' in globals():
            win_prob = model.predict_proba(new_game_features)[0]  # [勝率, 敗北率] の配列
            gunma_win_prob = win_prob[1]  # 群馬ニューフリーバーズの勝率
            opponent_win_prob = win_prob[0]  # 相手チームの勝率

            # 勝率を棒グラフで表示
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(["群馬ニューフリーバーズ", opponent_name], [gunma_win_prob, opponent_win_prob], color=['blue', 'red'])
            ax.set_ylabel("勝率")
            ax.set_ylim(0, 1)  # 確率なので0～1の範囲にする
            ax.set_title("試合結果の予測（勝率）")

            st.pyplot(fig)

            # 予測結果のテキスト表示
            prediction_result = "群馬ニューフリーバーズの勝利！" if prediction[0] == 1 else f"{opponent_name}の勝利！"
            st.markdown(f"### 予測結果: {prediction_result}")

        else:
            st.warning("モデルの準備ができていない、または試合データが不足しています。")
                      
               
       
    else:
        st.warning(
                   "データをアップロードしていない場合、左のサイドバーから必ず①から⑥の順にアップロードして下さい。\n\n"
                   "また既に①から⑥のデータを途中までアップロードしている場合は、⑥までの全てのデータをアップロードしてください。\n\n"
                   "※このページでは、全てのデータのアップロード後、即時に試合結果の予測を見る事が出来ます。")     

       





