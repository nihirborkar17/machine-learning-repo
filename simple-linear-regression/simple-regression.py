# Simple Linear Regression

#importing the libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
#importing dataset

df = pd.read_csv('simple-linear-regression/Salary_Data.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# splitting dataset into training and testing set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

# Training Simple linear regression model on training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the test set result
y_pred = regressor.predict(X_test)

# Visualising the training set result
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test Set result
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
