# This demonstrates end to end flow of a machine learning model
# 1. Split the data into features and labels
# 2. Filling/imputing: Disregarding missing values
# 3. Encoding: Encode the non-numerical values into numerical values

import pandas as pd
import numpy as np
import os




# First we demonstrate how to encode, hence we are picking up car-sales-extended version and not missing values version
car_sales = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '/data/car-sales-extended.csv')
print(car_sales.head())
print(car_sales.Make.dtype, car_sales.Colour.dtype, car_sales.Doors.dtype)

# 1. Split the data into features (X) and labels (y)
X = car_sales.drop('Price', axis=1)
y = car_sales['Price']

# Now since the Make, Colour are non-numerical data types, we first need to transform them
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
categorical_features = ['Make', 'Colour']
one_hot = OneHotEncoder()
transformer = ColumnTransformer([('one_hot', one_hot, categorical_features)], remainder='passthrough')

transformed_X = transformer.fit_transform(X)
print(f'Raw transformed Data:\n {transformed_X[:2]}')

print(f'Transformed Data in a Dataframe:\n {pd.DataFrame(transformed_X[:2])}')

# import RandomForestRegressor because we need to predict a numerical value
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

# Fit the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.2)
model.fit(X_train, y_train)
score = round(model.score(X_test, y_test) * 100, 2)
print(f'Score is: {score}')
