"""
Music streaming app or website , which takes users data
and then our machine learning model predicts the recommended songs for users
and show users the recommended songs for buying it
"""
# In This Machine Learning Model We Use Decision Tree ML Algorithm
# We Don't have to explicitly design this algorithm here
# It is already implemented in sklearn library of python 3
import pandas as pd
from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_dataframe = pd.read_csv('music.csv')
# Now we clean and prepare data
# by making input data set & output dataset
input_data = music_dataframe.drop(columns=['genre'])
output_data = music_dataframe.drop(columns=['age', 'gender'])
output_data2 = music_dataframe['genre']
input_data_train, input_data_test, output_data_train, output_data_test = train_test_split(input_data, output_data,
                                                                                          test_size=0.2)
dt_model = tree.DecisionTreeClassifier()
dt_model.fit(input_data, output_data)
prediction_data = dt_model.predict(input_data_test)
score = accuracy_score(output_data_test, prediction_data)
print("Model Accuracy : ", score)  # so our model accuracy score between 75% to 100%
# because we give 80% of input and output data for train the model
# 20% data input and output data for test the model
prediction_data = dt_model.predict([[30, 1]])
print(prediction_data)
# Decision Tree Graphical representation in dot file
tree.export_graphviz(dt_model, out_file="music_recommender.dot", feature_names=['age', 'gender'],
                     class_names=sorted(output_data2.unique()),
                     label='all', rounded=True, filled=True)
# OUR MODEL IS BASED ON DECISION TREE ALGORITHM OF MACHINE LEARNING

