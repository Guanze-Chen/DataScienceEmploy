import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('Zomato_df.csv')

df.drop('Unnamed: 0', axis=1, inplace=True)

print(df.head())
X = df.drop('rate', axis=1)
y = df['rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# Preparing Extra Tree Regression

ET_Model = ExtraTreesRegressor(n_estimators=120)
ET_Model.fit(X_train, y_train)

y_predict = ET_Model.predict(X_test)

import pickle
# save model to disk

pickle.dump(ET_Model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print(y_predict)
