# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

# Equipments Required:
Hardware – PCs

Software - Anaconda / Python 3.7 Installation / Moodle-Code Runner

# Algorithm

1.import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics.

10.Find the accuracy of our model and predict the require values.


Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: V JAIVIGNESH
RegisterNumber: 212220040055
*/
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

# OUTPUT

### Head

![image](https://github.com/neha074/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113016903/c39dcc17-a649-4f32-b3f7-602955e236ec)

### Info

![image](https://github.com/neha074/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113016903/dbbd37dc-8a59-4c7d-b42e-88478830a0f7)

### Identifying unwanted data

![image](https://github.com/neha074/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113016903/e6325b30-7265-4fdf-9106-59c9303c1487)

### value counts

![image](https://github.com/neha074/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113016903/094c07e3-6fcd-4710-8e5b-bd2a82a8efd0)

### salary head

![image](https://github.com/neha074/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113016903/f508e5eb-ee5d-4a0c-ad53-27892b31e068)

### x.head()

![image](https://github.com/neha074/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113016903/496f312e-be9f-4f9f-898b-7303ba3f013c)


![image](https://github.com/neha074/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113016903/e43cb1c0-b3d1-481c-831e-e46cbd75c3ed)

### Accuracy

![image](https://github.com/neha074/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113016903/d28bcab1-9e59-4b88-96b7-117e4f51cd38)

### Data Prediction

![image](https://github.com/neha074/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113016903/84b6edd1-7c22-411f-a532-15be78de2c2a)

# Result

Thus the program to implement the Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
