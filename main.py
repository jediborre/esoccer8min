import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


new_game = {
    'home_player': 'Flamingo',
    'away_player': 'Izzy',
    'res': 'W'
}
new_game = {
    'home_player': 'Bolec',
    'away_player': 'Izzy',
    'res': 'W'
}
new_game = {
    'home_player': 'Glumac',
    'away_player': 'Flamingo',
    'res': 'L'
}
new_game = {
    'home_player': 'Izzy',
    'away_player': 'Shone',
    'res': ''
}
new_game = {
    'home_player': 'Flamingo',
    'away_player': 'Bolec',
    'res': ''
}

outcome = {1: 'Home win', 0: 'Draw', -1: 'Away win'}

le = LabelEncoder()
data = pd.read_excel('bd.xlsx')
model_data = data[data['time'] == 'Full'].copy()
model_data['result'] = model_data.apply(
    lambda row: 1 if row['home_score'] > row['away_score'] else (
        -1 if row['home_score'] < row['away_score'] else 0
    ), axis=1
)

model_data['home_player_n'] = le.fit_transform(model_data['home_player']) # noqa
model_data['away_player_n'] = le.fit_transform(model_data['away_player']) # noqa

X = model_data[[
    'home_player_n',
    'away_player_n'
]]

y = model_data['result']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=400)  # Increase max_iter to 200
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

ew_game_df = pd.DataFrame([new_game])

new_game_df['home_player_n'] = le.transform(new_game_df['home_player'])
new_game_df['away_player_n'] = le.transform(new_game_df['away_player'])

X_new_game = new_game_df[['home_player_n', 'away_player_n']]
X_new_game_scaled = scaler.transform(X_new_game)

predicted_result = model.predict(X_new_game_scaled)

print(f"{new_game['home_player']} v {new_game['away_player']}: {outcome[predicted_result[0]]}")
