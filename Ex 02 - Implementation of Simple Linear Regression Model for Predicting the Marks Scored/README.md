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

Head and Tail for the dataframe

<img width="276" alt="Screenshot 2023-04-12 at 2 12 07 PM" src="https://user-images.githubusercontent.com/71516398/231403124-6ced8352-3c83-43a4-818c-2147cf3e8357.png">

Array value of X -\
<img width="276" alt="Screenshot 2023-04-12 at 2 12 17 PM" src="https://user-images.githubusercontent.com/71516398/231403193-297ac257-4436-47ec-96aa-17bd1951fbc9.png">

Array Value of Y -\
<img width="700" alt="Screenshot 2023-04-12 at 2 12 33 PM" src="https://user-images.githubusercontent.com/71516398/231403225-ed08b779-8ac8-4d0e-8ce9-5a28125558b0.png">

Values of Y prediction and Y test\
<img width="683" alt="Screenshot 2023-04-12 at 2 12 45 PM" src="https://user-images.githubusercontent.com/71516398/231403244-2d4efff3-9dfd-4067-ac72-602d70e38771.png">

Training Set Graph\
<img width="685" alt="Screenshot 2023-04-12 at 2 12 54 PM" src="https://user-images.githubusercontent.com/71516398/231403290-0b289bed-f8b4-404a-919c-8c14d03feba1.png">

Test Set Graph\
<img width="685" alt="Screenshot 2023-04-12 at 2 12 59 PM" src="https://user-images.githubusercontent.com/71516398/231403393-eaf5ecbb-15fc-4f74-8e16-32836a553c9a.png">

Values of MSE , MAE and RMSE

<img width="295" alt="Screenshot 2023-04-12 at 2 13 10 PM" src="https://user-images.githubusercontent.com/71516398/231403480-5b0d9b6a-8cbf-4c35-9479-d7895503fedd.png">


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

