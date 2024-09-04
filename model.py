import pandas as pd
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

players = [
    'Petruchio',
    'Bomb1to',
    'Kodak',
    'palkan',
    'Jekunam',
    'lion',
    'dm1trena',
    'Arcos',
    'Kray',
    'Inquisitor',
    'Koftovsky',
    'Boulevard',
    'Calvin',
    'Flamingo',
    'jAke',
    'Senior',
    'Sava',
    'Shone',
    'lowheels',
    'Bolec',
    'd1pseN',
    'Kravatskhelia',
    'nikkitta',
    'WBoy',
    'hotShot',
    'Izzy',
    'cl1vlind',
    'Glumac',
    'Galikooo',
    'Fratello',
    'Hotshot',
    'Wboy',
    'Menez',
    'SuperMario'
]


def is_player(player):
    global players
    return player in players


def get_game(new_game, le):

    df = pd.DataFrame([new_game])

    home_n = le.transform(df['home_player'])
    away_n = le.transform(df['away_player'])

    df['home_n'] = home_n
    df['away_n'] = away_n

    return df


def predict_game(models, le, scaler, new_game):
    resultados = {
        0: 'H',
        2: 'D',
        1: 'A'
    }

    game = get_game(new_game, le)

    x = game[['home_n', 'away_n']]
    x_scaled = scaler.transform(x)

    results = []
    for model in models:
        prediction = model.predict(x_scaled)
        results.append(resultados[prediction[0]])

    return results


def get_models():
    data = pd.read_excel('bd.xlsx')
    le = LabelEncoder()
    scaler = StandardScaler()

    model_data = data[data['time'] == 'Full'].copy()
    model_data['result'] = model_data.apply(
        lambda row: 0 if row['home_score'] > row['away_score'] else (
            2 if row['home_score'] < row['away_score'] else 1
        ), axis=1
    )

    model_data['home_n'] = le.fit_transform(model_data['home_player'])
    model_data['away_n'] = le.fit_transform(model_data['away_player'])

    x = model_data[['home_n', 'away_n']]
    y = model_data['result']
    X_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression(max_iter=400)  # Increase max_iter to 200
    model.fit(x_train, y_train)

    model_2 = RandomForestClassifier(n_estimators=100, random_state=42)
    model_2.fit(x_train, y_train)

    model_3 = GaussianNB()
    model_3.fit(x_train, y_train)

    model_4 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42) # noqa
    model_4.fit(x_train, y_train)

    model_5 = KNeighborsClassifier(n_neighbors=10)
    model_5.fit(x_train, y_train)

    model_6 = XGBClassifier(n_estimators=100, random_state=42)
    model_6.fit(x_train, y_train)

    model_7 = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model_7.fit(x_train, y_train)

    model_8 = SVC(kernel='rbf', random_state=42)
    model_8.fit(x_train, y_train)

    return [[
        model,
        model_2,
        model_3,
        model_4,
        model_5,
        model_6,
        model_7,
        model_8
    ], le, scaler]
