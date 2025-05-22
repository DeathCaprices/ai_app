from flask import Flask, render_template, request, redirect, session, url_for

app = Flask(__name__)
app.secret_key = 'secret-key'  # セッションに必要（適宜変更）

@app.route('/')
def index():
    players = session.get('players', [])
    return render_template('index.html', players=players)

@app.route('/add_player', methods=['POST'])
def add_player():
    player = {
        'name': request.form['name'],
        'avg': float(request.form.get('avg', 0)),
        'hr': int(request.form.get('hr', 0)),
        'obp': float(request.form.get('obp', 0)),
    }

    players = session.get('players', [])
    players.append(player)
    session['players'] = players
    return redirect(url_for('index'))

@app.route('/decide_order')
def decide_order():
    players = session.get('players', [])

    for p in players:
        p['score'] = p['avg'] * 3 + p['obp'] * 4 + p['hr'] * 2

    sorted_players = sorted(players, key=lambda x: x['score'], reverse=True)
    batting_order = [(i + 1, p['name'], p['avg'], p['hr'], p['obp']) for i, p in enumerate(sorted_players)]
    return render_template('order.html', batting_order=batting_order)

@app.route('/reset')
def reset():
    session.pop('players', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
