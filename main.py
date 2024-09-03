import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


df = pd.read_excel('bd.xlsx', engine='openpyxl')
df = df[df['time'] == 'Full']
df['winner'] = (df['home_score'] > df['away_score']).astype(int)
print(df[['home_score', 'away_score', 'winner']])


# le = LabelEncoder()
# df['home_player_encoded'] = le.fit_transform(df['home_player'])
# df['away_player_encoded'] = le.transform(df['away_player'])

# X = df[['home_player_encoded', 'away_player_encoded']]
# y = df['winner']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = LogisticRegression()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model accuracy: {accuracy:.2f}")
