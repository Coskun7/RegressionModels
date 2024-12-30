#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:44:28 2024

@author: mali
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = Salary_datasetcsv

x_train = df.YearsExperience.values.reshape(-1,1)
y_train = df.Salary.values.reshape(-1,1)

plt.scatter(x_train,y_train)
plt.show()

lin_reg = LinearRegression()


lin_reg.fit(x_train,y_train)

b0=lin_reg.intercept_
b1=lin_reg.coef_

def linear_regression_model(b0,b1,x):
    f_wb = b0 + b1 * x
    
    return f_wb
    
linear_regression_model(b0, b1, 17)

lin_reg.predict([[17]])

array = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])

y_test = linear_regression_model(b0, b1, array)

plt.scatter(x_train,y_train)
plt.plot(array,y_test.reshape(-1,1),c='r')
plt.show()

def cost_function (b0,b1,x,y):
    m=x.shape[0]
    cost=0
    for i in range(m):
        f_wb = b0 + b1 * x[i]
        cost += (f_wb - y[i])**2
    total_cost = (1 / ( 2 * m )) * cost
    
    return total_cost

