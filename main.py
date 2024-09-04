import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# from sklearn.metrics import accuracy_score, classification_report


def predict_game(new_game, model, le, scaler):
    # le = LabelEncoder()
    # scaler = StandardScaler()

    outcome = {2: 'H', 1: 'D', 0: 'A'}
    new_game_df = pd.DataFrame([new_game])

    home_n = le.transform(new_game_df['home_player'])
    away_n = le.transform(new_game_df['away_player'])

    new_game_df['home_player_n'] = home_n
    new_game_df['away_player_n'] = away_n

    X_new_game = new_game_df[[
        'home_player_n', 'away_player_n',
        # 'month', 'day',
        # 'hour', 'minute'
    ]]
    X_new_game_scaled = scaler.transform(X_new_game)

    predicted_result = model.predict(X_new_game_scaled)
    human_result = outcome[predicted_result[0]]

    return [
        human_result,
        home_n,
        away_n
    ]


def main():
    # id	fecha	time	score	home_score	away_score	home	home_player	away	away_player
    data = pd.read_excel('bd.xlsx')
    le = LabelEncoder()
    scaler = StandardScaler()

    model_data = data[data['time'] == 'Full'].copy()
    model_data['result'] = model_data.apply(
        lambda row: 2 if row['home_score'] > row['away_score'] else (
            0 if row['home_score'] < row['away_score'] else 1
        ), axis=1
    )

    model_data['fecha'] = pd.to_datetime(model_data['fecha'], format='%m/%d %H:%M')

    model_data['month'] = model_data['fecha'].dt.month
    model_data['day'] = model_data['fecha'].dt.day
    model_data['hour'] = model_data['fecha'].dt.hour
    model_data['minute'] = model_data['fecha'].dt.minute

    model_data['home_player_n'] = le.fit_transform(model_data['home_player']) # noqa
    model_data['away_player_n'] = le.fit_transform(model_data['away_player']) # noqa

    X = model_data[[
        'home_player_n',
        'away_player_n',
        # 'month', 'day',
        # 'hour', 'minute'
    ]]

    y = model_data['result']
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression(max_iter=400)  # Increase max_iter to 200
    model.fit(X_train, y_train)

    model_2 = RandomForestClassifier(n_estimators=100, random_state=42)
    model_2.fit(X_train, y_train)

    model_3 = GaussianNB()
    model_3.fit(X_train, y_train)

    model_4 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    model_4.fit(X_train, y_train)

    model_5 = KNeighborsClassifier(n_neighbors=10)
    model_5.fit(X_train, y_train)

    model_6 = XGBClassifier(n_estimators=100, random_state=42)
    model_6.fit(X_train, y_train)

    model_7 = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model_7.fit(X_train, y_train)

    model_8 = SVC(kernel='rbf', random_state=42)
    model_8.fit(X_train, y_train)

    juegos = [
        ['Flamingo', 'Shone', '1-5'],
        ['Sava', 'Calvin', '2-2'],
        ['Petruchio', 'jAke', '0-1'],
        ['Glumac', 'Bolec', '2-3'],
        ['jAke', 'palkan', '0-1'],
        ['Calvin', 'Petruchio', '1-2'],
        ['hotShot', 'Boulevard', '2-1'],
        ['Inquisitor', 'Kray', '2-2'],
        ['Kray', 'Kodak', '3-5'],
        ['Petruchio', 'Sava', '3-2'],
        ['palkan', 'Calvin', '4-3'],
        ['Boulevard', 'Inquisitor', '2-1'],
        ['Kodak', 'Boulevard', '2-2'],
        ['Calvin', 'jAke', '0-2'],
        ['Sava', 'palkan', '0-1'],
    ]
    juegos = [
        ['Calvin', 'Petruchio', '2-2'],
        ['hotShot', 'Boulevard', '1-1'],
        ['Inquisitor', 'Kray', '2-2'],
        ['Kray', 'Boulevard', '2-4'],
        ['hotShot', 'Inquisitor', '1-2'],
        ['Boulevard', 'Kodak', '1-1'],
        ['palkan', 'Sava', '1-3'],
    ]
    juegos = [
        ['Koftovsky', 'Kravatskhelia', ''],
        ['dm1trena', 'Wboy', ''],
        ['lion', 'Arcos', ''],
        ['Jekunam', 'd1pseN', ''],
        ['Bomb1to', 'Jekunam', ''],
        ['d1pseN', 'dm1trena', '']
    ]
    wins_1, wins_2, wins_3, wins_4, wins_5, wins_6, wins_7, wins_8 = 0, 0, 0, 0, 0, 0, 0, 0 # noqa
    loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, loss_8 = 0, 0, 0, 0, 0, 0, 0, 0 # noqa

    for juego in juegos:
        tmp = juego
        h = tmp[0]
        a = tmp[1]
        ft = tmp[2]
        s_h, s_a = '', ''
        model_results = {
            'H': 0,
            'D': 0,
            'A': 0
        }
        if ft:
            s_h, s_a = ft.split('-') if '-' in ft else [0, 0]
            s_h = int(s_h)
            s_a = int(s_a)

        result, h_n, a_n = predict_game({
            'home_player': h,
            'away_player': a,
            'month': 9,
            'day': 3,
            'hour': 15,
            'minute': 30
        }, model, le, scaler)

        result_2, h_n_2, a_n_2 = predict_game({
            'home_player': h,
            'away_player': a,
            'month': 9,
            'day': 3,
            'hour': 15,
            'minute': 30
        }, model_2, le, scaler)

        result_3, h_n_3, a_n_3 = predict_game({
            'home_player': h,
            'away_player': a,
            'month': 9,
            'day': 3,
            'hour': 15,
            'minute': 30
        }, model_3, le, scaler)

        result_4, h_n_4, a_n_4 = predict_game({
            'home_player': h,
            'away_player': a,
            'month': 9,
            'day': 3,
            'hour': 15,
            'minute': 30
        }, model_4, le, scaler)

        result_5, h_n_5, a_n_5 = predict_game({
            'home_player': h,
            'away_player': a,
            'month': 9,
            'day': 3,
            'hour': 15,
            'minute': 30
        }, model_5, le, scaler)

        result_6, h_n_6, a_n_6 = predict_game({
            'home_player': h,
            'away_player': a,
            'month': 9,
            'day': 3,
            'hour': 15,
            'minute': 30
        }, model_6, le, scaler)

        result_7, h_n_7, a_n_7 = predict_game({
            'home_player': h,
            'away_player': a,
            'month': 9,
            'day': 3,
            'hour': 15,
            'minute': 30
        }, model_7, le, scaler)

        result_8, h_n_8, a_n_8 = predict_game({
            'home_player': h,
            'away_player': a,
            'month': 9,
            'day': 3,
            'hour': 15,
            'minute': 30
        }, model_8, le, scaler)

        model_results[result] += 1
        model_results[result_2] += 1
        model_results[result_3] += 1
        model_results[result_4] += 1
        model_results[result_5] += 1
        model_results[result_6] += 1
        model_results[result_7] += 1
        model_results[result_8] += 1
        model_results['PH'] = (model_results['H']*100/8)
        model_results['PA'] = (model_results['A']*100/8)
        model_results['PD'] = (model_results['D']*100/8)

        if model_results['PH'] >= 75:
            result_9 = 'H'
        elif model_results['PA'] >= 75:
            result_9 = 'A'
        elif model_results['PD'] >= 75:
            result_9 = 'D'
        else:
            result_9 = '-'

        if result_9 != '-':
            if s_h:
                if s_h == s_a:
                    winner = 'D'
                elif s_h > s_a:
                    winner = 'H'
                else:
                    winner = 'A'

                if winner == result:
                    win_1 = 'SI'
                    wins_1 += 1
                else:
                    win_1 = 'NO'
                    loss_1 += 1

                if winner == result_2:
                    win_2 = 'SI'
                    wins_2 += 1
                else:
                    win_2 = 'NO'
                    loss_2 += 1

                if winner == result_3:
                    win_3 = 'SI'
                    wins_3 += 1
                else:
                    win_3 = 'NO'
                    loss_3 += 1

                if winner == result_4:
                    win_4 = 'SI'
                    wins_4 += 1
                else:
                    win_4 = 'NO'
                    loss_4 += 1

                if winner == result_5:
                    win_5 = 'SI'
                    wins_5 += 1
                else:
                    win_5 = 'NO'
                    loss_5 += 1

                if winner == result_6:
                    win_6 = 'SI'
                    wins_6 += 1
                else:
                    win_6 = 'NO'
                    loss_6 += 1

                if winner == result_7:
                    win_7 = 'SI'
                    wins_7 += 1
                else:
                    win_7 = 'NO'
                    loss_7 += 1

                if winner == result_8:
                    win_8 = 'SI'
                    wins_8 += 1
                else:
                    win_8 = 'NO'
                    loss_8 += 1

                print(f'{winner} {ft} | {result_6} {win_6} {result} {win_1} {result_2} {win_2} {result_3} {win_3} {result_4} {win_4} {result_5} {win_5} {result_7} {win_7} {result_8} {win_8} | {model_results["PH"]}% {model_results["PD"]}% {model_results["PA"]}% | {result_9}  | {h} v {a}') # noqa
            else:
                print(f'{ft} | {result_6} {result} {result_2} {result_3} {result_4} {result_5} {result_7} {result_8} | {model_results["PH"]}% {model_results["PD"]}% {model_results["PA"]}% | {result_9} | {h} v {a}') # noqa

    print('')
    print(f'XGBoost Classifier Wins: {wins_6}, Losses: {loss_6}')
    print(f'Logistic Regresion Wins: {wins_1}, Losses: {loss_1}')
    print(f'RandomForest Classifier Wins: {wins_2}, Losses: {loss_2}')
    print(f'Naive Bayes Wins: {wins_3}, Losses: {loss_3}')
    print(f'Simple multi-layer perceptron (MLP) neural network Wins: {wins_4}, Losses: {loss_4}')
    print(f'K-Nearest Neighbors (KNN) Wins: {wins_5}, Losses: {loss_5}')
    print(f'Gradient Boosting Classifier Wins: {wins_7}, Losses: {loss_7}')
    print(f'Support Vector Machine (SVM) Wins: {wins_8}, Losses: {loss_8}')


if __name__ == '__main__':
    main()
