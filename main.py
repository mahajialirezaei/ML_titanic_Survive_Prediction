import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


tit_df = pd.read_csv('titanic.csv')

CATEGORICAL_COLS = ['Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch']
NUMERIC_COLS = ['Age', 'Parch']

