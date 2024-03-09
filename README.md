# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:

**Step 1:** Import necessary libraries (NumPy, Matplotlib)

**Step 2:** Load the dataset

**Step 3:** Calculate the linear trend values using lLinearRegression Function.

**Step 4:** Calculate the polynomial trend values using PolynomialFeatures Function.

**Step 5:** End the program

### PROGRAM:
A - LINEAR TREND ESTIMATION
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
%matplotlib inline

train = pd.read_csv('AirPassengers.csv')
train['Month'] = pd.to_datetime(train['Month'], format='%Y-%m')
train['Year'] = train['Month'].dt.year
train.head()
year_data = train['Year'].values.reshape(-1, 1)
values_data = train['#Passengers'].values
plt.scatter(year_data, values_data, color='blue', label='Data Points')
plt.xlabel('Year')
plt.ylabel('Values')
plt.title('Before Performing Linear Trend')
plt.legend()
plt.show()

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(year_data, values_data)
intercept = model.intercept_
coefficients = model.coef_
print("Intercept: ",intercept, "Coefiicients:", coefficients)

new_years = np.array(train['Year']).reshape(-1, 1)
predicted_values = model.predict(new_years)

plt.scatter(year_data, values_data, color='blue', label='Data Points')
plt.plot(year_data, model.predict(year_data), color='red', label='Linear Regression Line')
plt.scatter(new_years, predicted_values, color='green', label='Predicted Values')
plt.xlabel('Year')
plt.ylabel('Values')
plt.title('Linear Regression Example')
plt.legend()
plt.show()
```
B- POLYNOMIAL TREND ESTIMATION
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from tabulate import tabulate
%matplotlib inline

train = pd.read_csv('AirPassengers.csv')
train['Month'] = pd.to_datetime(train['Month'], format='%Y-%m')
train['Year'] = train['Month'].dt.year
train.head()
x = train['Year'].values.reshape(-1, 1)
y = train['#Passengers'].values

poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)

y_pred = model.predict(x_poly)
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, y_pred, color='red', label='Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Trend Estimation')
plt.legend()
plt.show()
```
### OUTPUT

### A - LINEAR TREND ESTIMATION

#### Before Performing Linear Trend:
![image](https://github.com/Pavan-Gv/TSA_EXP2/assets/94827772/331f2085-5957-4987-bac8-332a1773d431)

#### After Performing Linear Trend:
![image](https://github.com/Pavan-Gv/TSA_EXP2/assets/94827772/d45bef37-d899-4019-9b78-b9a997b367c7)

### B- POLYNOMIAL TREND ESTIMATION

#### Before Performing Polynomial Trend:
![image](https://github.com/Pavan-Gv/TSA_EXP2/assets/94827772/30cdae80-5ec1-41ad-b801-789a7fb64cfb)

#### After Performing Polynomial Trend:
![image](https://github.com/Pavan-Gv/TSA_EXP2/assets/94827772/9fdb1438-c113-4745-981a-5117df85b6b5)

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
