# EX 03 Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import required pakages
2. read the dataset using pandas as a data frame
3. compute cost values 
4. Gradient Descent 

<img width="1030" alt="Screenshot 2023-04-02 at 10 07 35 PM" src="https://user-images.githubusercontent.com/71516398/229366463-e126f3ec-162c-4a0d-9571-02babb521222.png">

5.compute Cost function graph
6.compute prediction graph

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: V JAIVIGNESH
RegisterNumber:  212220040055
*/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("/content/ex1.csv")
data

#compute cost value
def computeCost(X,y,theta):
  m=len(y) 
  h=X.dot(theta) 
  square_err=(h - y)**2
  return 1/(2*m) * np.sum(square_err) 
  
 #computing cost value
data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m, 1)),data_n[:,0].reshape(m, 1),axis=1)
y=data_n[:,1].reshape (m,1) 
theta=np.zeros((2,1))
computeCost(X,y,theta) # Call the function

def gradientDescent (X,y, theta, alpha, num_iters):
  m=len (y)
  J_history=[]
  
  for i in range(num_iters):
    predictions = X.dot(theta)
    error = np.dot(X.transpose(), (predictions -y))
    descent=alpha * 1/m * error 
    theta-=descent
    J_history.append(computeCost (X,y, theta))
  return theta, J_history
  
  #h(x) value
theta,J_history = gradientDescent (X,y, theta, 0.01,1500)
print ("h(x) ="+str (round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"X1")

plt.plot(J_history)
plt.xlabel("Iteration") 
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data['a'],data['b'])
x_value=[x for x in range (25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value, color="r")
plt.xticks(np.arange (5,30,step=5)) 
plt.yticks(np.arange(-5,30,step=5)) 
plt.xlabel("Population of City (10,000s)") 
plt.ylabel("Profit ($10,000") 
plt.title("Profit Prediction")
# Text(0.5, 1.0, 'Profit Prediction')

def predict (x,theta):
# 11 11 11
# Takes in numpy array of x and theta and return the predicted value of y based on theta
  predictions= np.dot (theta.transpose (),x)
  return predictions[0]
  
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array ([1,7]), theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:

1.Compute Cost Value<br>
<img width="235" alt="Screenshot 2023-05-16 at 10 23 12 AM" src="https://github.com/JaivigneshJv/19AI410-Introduction-To-Machine-Learning/assets/71516398/0f9cd821-0780-4ca8-bdfa-2f423419d74f"><br>
2.h(x) Value<br>
<img width="235" alt="Screenshot 2023-05-16 at 10 23 16 AM" src="https://github.com/JaivigneshJv/19AI410-Introduction-To-Machine-Learning/assets/71516398/098d834f-b6b9-4dc3-a3db-ee5e37403d92"><br>
3.Cost function using Gradient Descent Graph<br>
<img width="717" alt="Screenshot 2023-04-02 at 9 47 46 PM" src="https://user-images.githubusercontent.com/71516398/229365976-528d32b4-9600-4d03-b88e-ab72feb36768.png"><br>
4.Profit Prediction Graph<br>
<img width="614" alt="Screenshot 2023-05-16 at 10 30 31 AM" src="https://github.com/JaivigneshJv/19AI410-Introduction-To-Machine-Learning/assets/71516398/9b5c094b-7742-4e0d-bcd9-795fbfdb90b2"><br>
5.Profit for the Population 35,000<br>
<img width="480" alt="Screenshot 2023-05-16 at 10 23 28 AM" src="https://github.com/JaivigneshJv/19AI410-Introduction-To-Machine-Learning/assets/71516398/bd91d309-eafd-4835-94d0-0c730f955cfb"><br>
6.Profit for the Population 70,000<br>
<img width="480" alt="Screenshot 2023-05-16 at 10 23 33 AM" src="https://github.com/JaivigneshJv/19AI410-Introduction-To-Machine-Learning/assets/71516398/30ba29c5-e6b2-4017-add0-c35dc540c58e"><br>





## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
