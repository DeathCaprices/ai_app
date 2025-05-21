import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ダミーデータ: [風速, 傾斜, クラブの種類]
X = np.array([
    [3, 5, 1],
    [5, 0, 2],
    [0, -3, 3],
    [10, 2, 1],
])

# 飛距離
y = np.array([200, 250, 180, 210])

# スケーリング
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# モデルの定義と学習（LinearRegression）
model = LinearRegression()
model.fit(X_scaled, y)

# モデルの保存（scikit-learnの場合）
with open('shot_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# スケーラーの保存
with open('shot_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
