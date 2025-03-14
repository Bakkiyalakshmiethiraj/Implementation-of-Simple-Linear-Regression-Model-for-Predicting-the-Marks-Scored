# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Gather data consisting of two variables. Input- a factor that affects the marks and Output - the marks scored by students
2. Plot the data points on a graph where x-axis represents the input variable and y-axis represents the marks scored
3. Define and initialize the parameters for regression model: slope controls the steepness and intercept represents where the line crsses the y-axis
4. Use the linear equation to predict marks based on the input Predicted Marks = m.(hours studied) + b
5. for each data point calculate the difference between the actual and predicted marks
6. Adjust the values of m and b to reduce the overall error. The gradient descent algorithm helps update these parameters based on the calculated error
7. Once the model parameters are optimized, use the final equation to predict marks for any new input data

## Program:

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Bakkiyalakshmi E
RegisterNumber:  212223220012
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset=pd.read_csv('/content/student_scores.csv')
print(dataset.head())
print(dataset.tail())

dataset.info()

X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

mse=mean_squared_error(Y_test,Y_pred)
print("MSE =",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE: ',mae)
rmse=np.sqrt(mse)
print("RMSE",rmse)

plt.scatter(X_test,Y_test,color="green")
plt.plot(X_test,Y_pred,color="black")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


a=np.array([[13]])
ans=reg.predict(a)
print(ans)

```

## Output:

 # Head Values:
 ![image](https://github.com/user-attachments/assets/296fd908-c78c-4bed-9fe3-a21834b9de8f)
 # Tail Value:
 ![image](https://github.com/user-attachments/assets/32e4cafc-d5d3-42d2-aaf4-4bed8a296f7e)
 # X and Y Values:
 ![image](https://github.com/user-attachments/assets/846dd26d-d3a4-407a-b678-0dc59e741c55)
 # Y_pred,Y_test,X_train.shape,X_test.shape,Y_train.shape,Y_test.shape:
 ![image](https://github.com/user-attachments/assets/8571ea7f-75bb-43db-9829-e4cc2c41dd6c)
 # MSE,MAE,RMSE:
 ![image](https://github.com/user-attachments/assets/18a5e28c-8daf-41d3-95a5-f541d6806a7a)
 # Testing test:
 ![image](https://github.com/user-attachments/assets/4c43f5b1-15c7-4e97-a43a-df79198bae01)
 # array[13]:
 ![image](https://github.com/user-attachments/assets/966812c7-4e9f-4fdb-b9a3-a86e8dedc84a)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
