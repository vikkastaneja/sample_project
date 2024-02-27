# This demonstrates end to end flow of a machine learning model
# 1. Split the data into features and labels
# 2. Filling or imputing: Refilling or disregarding missing values
# 3. Encoding: Encode the non-numerical values into numerical values

import os

from CommonFunctionality import run_all_car_sales

data_file = os.path.dirname(os.path.abspath(__file__)) + '/' + 'data/car-sales-extended.csv'

run_all_car_sales(data_file)
