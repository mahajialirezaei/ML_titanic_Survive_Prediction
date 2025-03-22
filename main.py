import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


tit_df = pd.read_csv('titanic.csv')

CATEGORICAL_COLS = ['Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch']
NUMERIC_COLS = ['Age', 'Parch']

tit_df.dropna(subset=['Survive', 'Age', 'Fare', 'Embarked'], inplace=True)

y = tit_df['Survive']
x = pd.get_dummies(tit_df[CATEGORICAL_COLS + NUMERIC_COLS])
sc = StandardScaler()

x[NUMERIC_COLS] = sc.fit_transform(x[NUMERIC_COLS])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=63)


lin_model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(x_train.shape[1],)),
])

lin_model.compile(
   optimizer='adam',
    loss='binary_crossentropy',
    matrics='accuracy'
)

lin_model.fit(x_train, y_train, epochs=25, batch_size=32, validation_split=0.2)