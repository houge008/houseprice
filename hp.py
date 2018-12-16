# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 19:37:20 2018
House Price predicting from kaggle

The step in this work are:
  1.Build a Random Forest model with all the traindata(X,y)
  2.Read in the "test"data,which does not inlucde values 
  for the target.Predict home values in the test data with
  the Random Forest model
  3.Submit those predictions to the competition and see score
  4.Optionally,come back to see if you can improve the your model
  by adding features or changing the model.Then I can resubmiiit to 
  see how that stacks up on the competiion leaderboard.
  
@author:Hou Dongjie
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

home_data = pd.read_csv('train.csv')
#create target object and call it y
y = home_data.SalePrice
#create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features] #select the features to build model

#Split into validation and training data
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=1)
#Special Model DTR
my_model = DecisionTreeRegressor(random_state=1)
#Fit model
my_model.fit(train_X,train_y)

#Make validation predictions and calculate mean absolute error
val_predictions = my_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions,val_y)
print("Validation MAE when not specifying max_leaf_nodes:{:,.0f}".format(val_mae))

#Using best value for max_leaf_nodes
my_model = DecisionTreeRegressor(max_leaf_nodes=100,random_state=1)
my_model.fit(train_X,train_y)
val_predictions = my_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions,val_y)
print("Validation MAE for best value of max_leaf_nodes:{:,.0f}".format(val_mae))

#Define the model rf. Set random_state=1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X,train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions,val_y)
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

"""
To improve accuracy, craete a new Random Forest model,
which I will trian on all training data ,set random_state=0
"""
rf_model_full_data = RandomForestRegressor(random_state=0)
rf_model_full_data.fit(train_X,train_y)
rf_full_predictions = rf_model_full_data.predict(val_X)
rf_mae_full_data = mean_absolute_error(rf_full_predictions,val_y)
print("Validation MAE on full data for Random Forest Model: {:,.5f}".format(rf_mae_full_data))

"""
MAKE PREDICTIONS:read the file of "test.data",
and apply my rf_full model to make predictions
"""
test_data = pd.read_csv('test.csv')
# Create test_X which comes from test_data but includes only the columes I used for prediction.
#The list of columes is stored in a variable called features
test_X = test_data[features]

#make predictions which we will submit
test_preds = rf_model_full_data.predict(test_X)

#The line below shows how to save my data in the format needed to score it in the kaggle competition
output = pd.DataFrame({'ID':test_data.Id,'SalePrice':test_preds})
output.to_csv("submission.csv",index=False)