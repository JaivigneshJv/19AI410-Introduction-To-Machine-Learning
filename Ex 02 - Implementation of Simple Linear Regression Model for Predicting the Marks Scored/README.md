# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python.
2. Set variables for assigning data set values.
3. Import Linear Regression from the sklearn.
4. Assign the points for representing the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtain the LinearRegression for the given data.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: V JAIVIGNESH
RegisterNumber:  212220040055
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('/content/student_scores.csv')

data.head()

data.tail()

x=data.iloc[:,:-1].values  
y=data.iloc[:,1].values

print(x)
print(y)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0 )

regressor=LinearRegression() 
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test) 
print(y_pred)

print(y_test)

#for train values
plt.scatter(x_train,y_train) 
plt.plot(x_train,regressor.predict(x_train),color='black') 
plt.title("Hours Vs Score(Training set)") 
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()

#for test values
y_pred=regressor.predict(x_test) 
plt.scatter(x_test,y_test) 
plt.plot(x_test,regressor.predict(x_test),color='black') 
plt.title("Hours Vs Score(Test set)") 
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()

import sklearn.metrics as metrics

mae = metrics.mean_absolute_error(x, y)
mse = metrics.mean_squared_error(x, y)
rmse = np.sqrt(mse)  

print("MAE:",mae)
print("MSE:", mse)
print("RMSE:", rmse)

```

## Output:
<img width="576" alt="Screenshot 2023-04-02 at 10 43 39 PM" src="https://user-images.githubusercontent.com/71516398/229368293-cfbc914c-05a0-43a1-8d3e-17b4d343a57f.png">
<img width="576" alt="Screenshot 2023-04-02 at 10 43 45 PM" src="https://user-images.githubusercontent.com/71516398/229368298-0f6e29a6-7d8a-441a-ba15-25a6fc0a4bed.png">
<img width="409" alt="Screenshot 2023-04-02 at 10 43 58 PM" src="https://user-images.githubusercontent.com/71516398/229368300-6033ba84-9b65-4d05-b757-d5698aad8de8.png">


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

## Colab:
https://colab.research.google.com/drive/1Cz2Fudz4rbt0aXKGcWAr4-AXxbwJku_4?usp=sharing
