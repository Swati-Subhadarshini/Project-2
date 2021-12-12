# Project-2

Project on Machine Learning

## King County Housing Price Prediction

The goal of this FinTech analysis is to use machine learning in order to help us understand the relationship between house features and how its variables are used to predict house price.

We planned to predict house prices using three machine learning models, in terms of minimizing the difference between predicted and actual prices.



Data used: King County Home Sales between May 2014 - May 2015.
Kaggle-kc_house_data.


## Technologies Used

Leveraging Jupyter Notebook, Python 3.5+.
Leveraging the use of Pandas which is included in Anaconda.
Please go to: https://docs.anaconda.com/anaconda/install/ - For finding the proper installation guide.
The packages will be:
Gitbash CLI is used to pull and push the code from local repository to remote repository (GitHub).
Code written with the help of Jupyter Notebook

### Instructions:

This project is done by comparing three models (Multiple Regression, Neural Networking and Gradient Boost Algorithm)

## Multiple Regression Model:
This project consists of the following subsections:

## Libraries & Dependencies:

import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from pandas.tseries.offsets import DateOffse
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import ensemble
import warnings
warnings.filterwarnings('ignore')

### Read the csv file into the dataframe.
` kc_house_df = pd.read_csv("kc_house_data.csv")
Review the dataframe
kc_house_df.head()`

### check if there are any Null values
`kc_house_df.isnull().sum()`

### drop some unnecessary columns
`kc_house_df = kc_house_df.drop(columns=['date', 'id', 'zipcode'], axis=1)
kc_house_df.head()`

Separate the data into labels and features
`y = kc_house_df["price"]`

Separate the X variable, the features
`X = kc_house_df.drop(columns=["price"])`

Split the data using train_test_split
Assign a random_state of 1 to the function
`Import the train_test_learn module
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)`

Use scikit-learn’s StandardScaler to standardize the numerical features.
`from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)`

# Fit the scaler to the features training dataset
`X_train = X_scaler.fit_transform(X_train.astype(np.float))
X_test = X_scaler.transform(X_test.astype(np.float))
Create a Multiple Regression Model with the Original Data`

Create a Multiple Regression Model with the Original Data Step 1: Fit a multiple regression model by using the training data (X_train and y_train).
Predict a Multiple Regression Model with Training Data
Split the Data into Training and Testing Sets Open the starter code notebook and then use it to complete the following steps.
Import the LogisticRegression module from SKLearn

`from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr_model = lr.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)`


Evaluate the model’s performance by doing the following:

1 Evaluate the performance of the algorithm (MAE - MSE - RMSE)
2 Mean absolute error (MAE) is a measure of errors between paired observations expressing the same phenomenon. 
3 Examples of Y versus X include comparisons of predicted versus observed.
4 Explained_variance_score: variance of prediction errors and actual values - Best possible score is 1.0; lower values are less accurate predictors.


## Neural Networking

MSE is the loss function, the adam optimizer, and the additional metric accuracy.

Fit the model with the training data, using 10000 epochs.

Plot the model’s loss function and accuracy over the 10000 epochs.

Evaluate the model using testing data and the evaluate method. Using one hidden layer with 10 nodes on it.

## Code:

Define the model - deep neural net with two hidden layers. Define the number of hidden nodes for the model. Create the Sequential model instance.  Add a Dense layer specifying the number of inputs, the number of hidden nodes, and the activation function. # Add the output layer to the model specifying the number of output neurons and activation function.

`number_inputs = 16
number_hidden_nodes = 12
neuron = Sequential()
neuron.add(Dense(units=number_hidden_nodes, input_dim=number_inputs, activation="relu"))
neuron.add(Dense(1, activation="linear"))``

##### Compile the Sequential model. Fit the model using 1000 epochs and the training data

model = neuron.fit(X_train_scaled, y_train, epochs=10000)
neuron.compile(loss="mean_squared_error", optimizer="adam", metrics=["mse"])`

# Create a DataFrame using the model history and an index parameter

`model_plot = pd.DataFrame(model.history, index=range(1, len(model.history["loss"]) + 1))`
`y_pred = neuron.predict(X_test_scaled)`

# Set the model's file path
# Export your model to an HDF5 file

`file_path = Path("Resources/house_price_prediction_Model1.h5")
neuron.save(file_path)`
`
### Gradient Boost

gbModel = ensemble.GradientBoostingRegressor(n_estimators = 100, max_depth = 3, min_samples_split = 20,
          learning_rate = 0.1, loss='ls')
gbModel.fit(X_train_scaled, y_train)
gbModel.score(X_test_scaled,y_test)

#Calculate Mean Absolute Error

y_pred = gbModel.predict(X_test_scaled)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

## Contributors

This project is designed by - 

Swati Subhadarshini
Emal id: sereneswati@gmail.com

LinkedIn link: www.linkedin.com/in/swati-subhadarshini

Somaye Nargesi
Email id: srn1358@gmail.com
 

Hiep Le
Email id: hieple@uw.edu
