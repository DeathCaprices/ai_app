from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# モデルの読み込み
with open('shot_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        wind = float(request.form['wind'])
        slope = float(request.form['slope'])
        club = int(request.form['club'])  # 1=Driver, 2=Iron, etc.

        features = np.array([[wind, slope, club]])
        prediction = model.predict(features)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
