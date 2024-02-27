import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# from colorama import init as colorama_init
# from colorama import Fore
# from colorama import Style

class bcolors:
    HEADER = '\33[95m'
    OKBLUE = '\33[94m'
    OKCYAN = '\33[96m'
    OKGREEN = '\33[92m'
    WARNING = '\33[93m'
    FAIL = '\33[91m'
    ENDC = '\33[0m'
    BOLD = '\33[1m'
    UNDERLINE = '\33[4m'
    BLINK = '\33[5m'

def run_all_car_sales(absolute_path_file_name):
    # colorama_init()
    if os.path.isfile(absolute_path_file_name) == False:
        exit(f'{bcolors.BLINK}{bcolors.FAIL}****** File \'{absolute_path_file_name}\' does not exist ******{bcolors.ENDC}')

    print(f'File used: {bcolors.OKGREEN}{absolute_path_file_name}{bcolors.ENDC}')
    # To get predictable results, setting random seed
    np.random.seed(80)

    # First we demonstrate how to encode, hence we are picking up car-sales-extended version and not missing values version
    car_sales = pd.read_csv(absolute_path_file_name)
    print(car_sales.head())
    print(f'Car Make data type: {car_sales.Make.dtype}\nCar Colour data type: {car_sales.Colour.dtype}\nCar Doors data type: {car_sales.Doors.dtype}')

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

    # Although Scikit-Learn latest revision automatically handles the NaNs in the data, there are multiple ways to handle manually
    # Option 1. 