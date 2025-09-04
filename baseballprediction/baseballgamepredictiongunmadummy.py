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
import streamlit as st



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


    else:
    # サンプルデータ
        data = {
            "試合": [14],
            "勝": [8],
            "負": [3],
            "引分": [3],
            "打率": [0.268],
            "安打": [149],
            "本塁打": [10],
            "打点": [71],
            "盗塁": [5],
            "犠打": [18],
            "四死球": [63],
            "三振": [51],
            "失策": [5],
            "完投": [3],
            "投球回": ["172 2/3"],
            "被安打": [102],
            "被本塁打": [7],
            "与四死球": [44],
            "奪三振": [171],
            "失点": [31],
            "自責点": [30],
            "防御率": [1.56],
            "球数": [2546]
            }

        df1 = pd.DataFrame(data)
        # 全ての小数点を第3位まで丸める   
        df1 = df1.round(3)  
        st.markdown("##### 群馬ニューフリーバーズ昨年の総合データ(サンプルデータを使用してます。)")
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


    else:
        # サンプルデータ
        data = {
            "試合": [12],
            "勝": [8],
            "負": [4],
            "引分": [0],
            "打率": [0.326],
            "安打": [128],
            "本塁打": [8],
            "打点": [49],
            "盗塁": [19],
            "犠打": [17],
            "四死球": [57],
            "三振": [65],
            "失策": [7],
            "完投": [3],
            "投球回": ["168 2/3"],
            "被安打": [144],
            "被本塁打": [13],
            "与四死球": [72],
            "奪三振": [111],
            "失点": [63],
            "自責点": [57],
            "防御率": [3.04],
            "球数": [2602]
                }

        df2 = pd.DataFrame(data)
        # 全ての小数点を第3位まで丸める   
        df2 = df2.round(3)  
        st.markdown(f"##### {opponent_name}昨年の総合データ(サンプルデータを使用してます。)")
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

    else:
        # サンプルデータ
        data = {
            "氏名": ["熊谷", "宗山", "尾瀬", "石郷岡", "印出", "齋藤大", "松下", "中山", "小島河"],
            "試合": [11, 13, 13, 13, 13, 12, 14, 12, 13],
            "打席": [44, 60, 61, 47, 60, 47, 64, 45, 53],
            "打数": [34, 50, 49, 33, 50, 39, 54, 41, 50],
            "得点": [7, 10, 11, 7, 9, 7, 8, 4, 6],
            "安打": [16, 20, 19, 12, 18, 14, 19, 14, 17],
            "二塁打": [2, 3, 3, 3, 5, 3, 3, 1, 3],
            "三塁打": [1, 0, 0, 0, 0, 0, 0, 0, 3],
            "本塁打": [0, 2, 0, 0, 0, 1, 5, 1, 1],
            "塁打": [20, 29, 22, 15, 23, 20, 37, 18, 29],
            "打点": [5, 12, 5, 6, 9, 5, 13, 4, 12],
            "盗塁": [1, 0, 0, 2, 1, 0, 0, 1, 0],
            "犠打": [6, 1, 1, 2, 0, 3, 2, 1, 2],
            "四死球": [4, 9, 11, 12, 10, 5, 8, 3, 1],
            "三振": [6, 1, 6, 5, 7, 7, 6, 8, 5],
            "失策": [1, 1, 0, 0, 0, 1, 1, 1, 0],
            "打率": [0.471, 0.400, 0.388, 0.364, 0.360, 0.359, 0.352, 0.341, 0.340]
                }
        df3 = pd.DataFrame(data)
        # 全ての小数点を第3位まで丸める
        df3 = df3.round(3)  
        st.markdown("##### 群馬ニューフリーバーズ打撃データ(サンプルデータを使用してます。)")
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

    else:
        # サンプルデータ
        data = {
        "氏名": ["前田", "水鳥", "姫木", "中津", "直井", "藤森康", "榊原", "小澤", "鈴木唯"],
        "試合": [13, 13, 11, 14, 13, 14, 13, 13, 14],
        "打席": [55, 56, 44, 66, 64, 55, 45, 54, 53],
        "打数": [45, 52, 40, 53, 51, 51, 38, 47, 41],
        "得点": [7, 7, 0, 9, 6, 7, 7, 7, 6],
        "安打": [15, 17, 13, 17, 16, 15, 11, 13, 11],
        "二塁打": [5, 2, 2, 2, 3, 3, 1, 2, 2],
        "三塁打": [2, 0, 0, 0, 1, 0, 1, 0, 2],
        "本塁打": [1, 2, 0, 2, 0, 1, 2, 0, 0],
        "塁打": [27, 25, 15, 25, 21, 21, 20, 15, 17],
        "打点": [7, 6, 4, 8, 4, 4, 10, 3, 3],
        "盗塁": [0, 5, 0, 3, 2, 2, 1, 3, 3],
        "犠打": [3, 2, 0, 2, 3, 1, 2, 2, 2],
        "四死球": [7, 2, 4, 11, 10, 3, 5, 5, 10],
        "三振": [7, 12, 6, 7, 4, 11, 8, 2, 8],
        "失策": [1, 4, 0, 0, 0, 0, 0, 1, 1],
        "打率": [0.333, 0.327, 0.325, 0.321, 0.314, 0.294, 0.289, 0.277, 0.268]
            }
        df4 = pd.DataFrame(data)  
        # 小数点をすべて第3位まで丸める
        df4 = df4.round(3)   
        st.markdown(f"##### {opponent_name}打撃データ(サンプルデータを使用してます。)")
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

    else:
    # サンプルデータ
        data = {
        "氏名": ["渡辺和", "毛利", "伊藤樹", "宮城"],
        "試合": [9, 6, 8, 5],
        "完投": [2, 0, 1, 0],
        "完了": [3, 0, 0, 0],
        "当初": [4, 6, 7, 5],
        "無点勝": [1, 0, 0, 0],
        "無四球": [0, 0, 0, 0],
        "勝利": [3, 3, 6, 1],
        "敗戦": [2, 1, 1, 1],
        "引分": [0, 0, 0, 0],
        "打者": [201, 111, 233, 113],
        "投球回": ["54.0", "29 1/3", "60.0", "29 1/3"],
        "被安打": [30, 17, 36, 19],
        "被本塁打": [1, 2, 4, 0],
        "与四死球": [11, 9, 17, 7],
        "奪三振": [57, 28, 60, 26],
        "失点": [8, 5, 12, 6],
        "自責点": [7, 5, 12, 6],
        "防御率": [1.17, 1.53, 1.80, 1.84],
        "球数": [743, 451, 874, 478]
            }

        df5 = pd.DataFrame(data)
        # 小数点をすべて第2位まで丸める
        df5 = df5.round(2)         
        st.markdown("##### 群馬ニューフリーバーズ投手データ(サンプルデータを使用してます。)")
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

    else:
        # サンプルデータ
        data = {
        "氏名": ["篠木", "竹中", "小畠", "渡辺"],
        "試合": [8, 8, 7, 6],
        "完投": [1, 0, 0, 2],
        "完了": [0, 1, 0, 0],
        "当初": [7, 3, 7, 4],
        "無点勝": [0, 0, 0, 0],
        "無四球": [0, 0, 0, 0],
        "勝利": [3, 2, 1, 1],
        "敗戦": [2, 1, 2, 4],
        "引分": [0, 0, 0, 0],
        "打者": [240, 125, 182, 161],
        "投球回": ["59","30 2/3","42 2/3","36 1/3"],
        "被安打": [45, 24, 40, 35],
        "被本塁打": [4, 2, 2, 5],
        "与四死球": [22, 12, 17, 21],
        "奪三振": [52, 22, 20, 17],
        "失点": [17, 10, 17, 19],
        "自責点": [17, 10, 15, 15],
        "防御率": [2.59, 2.93, 3.16, 3.72],
        "球数": [937, 437, 646, 582],
                }
        df6 = pd.DataFrame(data)
        # 小数点をすべて第2位まで丸める
        df6 = df6.round(2)        
        st.markdown(f"##### {opponent_name}投手データ(サンプルデータを使用してます。)")
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
        ax[0].set_title("両チームの昨年の打撃データ", fontproperties=font_prop)
        ax[0].set_xlabel("群馬ニューフリーバーズ", fontproperties=font_prop)
        ax[0].set_ylabel(f"{opponent_name}", fontproperties=font_prop)
        ax[0].legend(prop=font_prop)
        
        
        

        # 各特徴量ごとにプロット（色を変える）
        ax[1].scatter(df1["被安打"], df2["被安打"], label="被安打", color="blue")
        ax[1].scatter(df1["被本塁打"], df2["被本塁打"], label="被本塁打", color="red")
        ax[1].scatter(df1["与四死球"], df2["与四死球"], label="与四死球", color="green")
        ax[1].scatter(df1["奪三振"], df2["奪三振"], label="奪三振", color="orange")
        ax[1].scatter(df1["失点"], df2["失点"], label="失点", color="purple")
        ax[1].scatter(df1["自責点"], df2["自責点"], label="自責点", color="brown")
        
        # 軸ラベルと凡例
        ax[1].set_title("両チームの昨年の投手データ", fontproperties=font_prop)
        ax[1].set_xlabel("群馬ニューフリーバーズ", fontproperties=font_prop)
        ax[1].set_ylabel(f"{opponent_name}", fontproperties=font_prop)
        ax[1].legend(prop=font_prop)
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
        ax_bar.set_xticklabels(teams,fontproperties=font_prop)
        ax_bar.set_ylabel("値", fontproperties=font_prop)
        ax_bar.set_title("両チームの昨年の防御率と打率の比較",fontproperties=font_prop)
        ax_bar.legend(prop=font_prop)


        
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
        ax[0].set_title("両チームの直近の打撃データ",fontproperties=font_prop)
        ax[0].set_xlabel("群馬ニューフリーバーズ",fontproperties=font_prop)
        ax[0].set_ylabel(f"{opponent_name}",fontproperties=font_prop)
        ax[0].legend(prop=font_prop)
        
        

        # 各特徴量ごとにプロット（色を変える）
        ax[1].scatter(df5["被安打"].sum(), df6["被安打"].sum(), label="被安打", color="blue")
        ax[1].scatter(df5["被本塁打"].sum(), df6["被本塁打"].sum(), label="被本塁打", color="red")
        ax[1].scatter(df5["与四死球"].sum(), df6["与四死球"].sum(), label="与四死球", color="green")
        ax[1].scatter(df5["奪三振"].sum(), df6["奪三振"].sum(), label="奪三振", color="orange")
        ax[1].scatter(df5["失点"].sum(), df6["失点"].sum(), label="失点", color="purple")
        ax[1].scatter(df5["自責点"].sum(), df6["自責点"].sum(), label="自責点", color="brown")
        
        # 軸ラベルと凡例
        ax[1].set_title("両チームの直近の投手データ",fontproperties=font_prop)
        ax[1].set_xlabel("群馬ニューフリーバーズ",fontproperties=font_prop)
        ax[1].set_ylabel(f"{opponent_name}",fontproperties=font_prop)
        ax[1].legend(prop=font_prop)
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
        ax_bar.set_xticklabels(teams,fontproperties=font_prop)
        ax_bar.set_ylabel("値",fontproperties=font_prop)
        ax_bar.set_title("両チームの直近の防御率と打率の比較",fontproperties=font_prop)
        ax_bar.legend(prop=font_prop)


        
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
            ax.bar([0, 1], [gunma_win_prob, opponent_win_prob], color=['blue', 'red'])
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["群馬ニューフリーバーズ", opponent_name], fontproperties=font_prop)

            ax.set_ylabel("勝率",fontproperties=font_prop)
            ax.set_ylim(0, 1)  # 確率なので0～1の範囲にする
            ax.set_title("試合結果の予測（勝率）",fontproperties=font_prop)


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

       





