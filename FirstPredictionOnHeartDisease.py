# In this sample program, we will generate a prediction model that is most accurate,
# save it and retrieve it and compare the score.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

# Read the records from heart-disease.csv file and create a data frame from it
heart_disease = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + "/data/heart-disease.csv")
# print(type(heart_disease))
print("Heard Disease record count: " + str(heart_disease.size))

# Split the data from skikit-learn library, between features (inputs) and labels (outputs)
X = heart_disease.drop('target', axis=1) # Features
y = heart_disease['target'] # Label

# Chose the model with default parameters
model = RandomForestClassifier() # model.get_params()
print('Model n_estimators: ' + str(model.get_params()['n_estimators']))
# Train the model (aka fit the model) using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train) # X_train has training features, y_train has training labels
                            # X_test has test features, y_test has test labels

# print('Train feature')
# print(X_train.head())
# print(X_test.head())

# print('Test feature')
# print(y_train.head())
# print(y_test.head())

# print(heart_disease.head())

# Now predict. Note that the prediction results are stored within model
# That's why when the stored model is loaded and score is retrieved w.r.t. X_test, it gives the same results
model.predict(X_test) 

# print(type(y_predicitions))
# print(y_predicitions.shape)
# print(type(X_test))
score = model.score(X_test, y_test)
print(f'Score: {score * 100:.2f}')
# print(y_test.head())

# 5. Improve a model
# Try different amount of n_estimators
np.random.seed(42)
(max_score, max_estimator, max_model) = (0, 0, 0)
for i in range(10, 100, 10):
    print(f"Trying model with {i} estimators...")
    clf = RandomForestClassifier(n_estimators=i).fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    if max_score < score:
        max_score = score
        max_estimator = i
        max_model = clf
    print(f"Model accuracy on test set: {score * 100:.2f}%")
    print("")

print(f'max score: {max_score * 100:.2f}, max estimator: {max_estimator}%')

# 6. Save a model and load it
import pickle

pickle.dump(max_model, open("max_model.pkl", "wb"))
loaded_model = pickle.load(open("max_model.pkl", "rb"))
print(f"Saved model score: {loaded_model.score(X_test, y_test)* 100:.2f}")