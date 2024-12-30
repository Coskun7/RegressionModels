#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:07:54 2024

@author: mali
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)
from sklearn.linear_model import LinearRegression

#%%Single Linear Regression 
# f_wb = w1 * x1 + b
# cost function = 1/2m (i=1 ∑ m) (f_wb - y)**2

df=pd.read_csv('/Users/mali/Downloads/Salary_dataset.csv')

x=df.iloc[:,1].to_numpy()
y=df.iloc[:,2].to_numpy()

x_train=x.reshape(-1,1)
y_train=y.reshape(-1,1)

w1=500
b1=25000

def simple_lin_model(x,w,b):
    """
    

    Parameters
    ----------
    x : (scalar) years experience
    w : scalar coefficient of x
    b : constant

    Returns
    -------
    pred : Salary of yearly

    """
    pred = x * w + b
    
    return pred 

def compute_cost(x,y,w,b):
    m=x.shape[0]
    cost=0
    for i in range(m):
        f_wb= w * x[i]+ b
        cost = cost + (f_wb - y[i])**2
    
    total_cost = 1/(2*m) * cost
    
    return total_cost

print(f'when w = 500 and b=25000 our cost is : {compute_cost(x_train,y_train,w1,b1)}')

# we using gradient decsent for find w,b values 
# gradient descent for w = w - alpha * 1/m (i=1 ∑ m) (f_wb[i] - y[i])*x[i]
# gradient decsent for b = b - alpha * 1/m (i=1 ∑ m) f-wb[i] - y[i]

def compute_gradient(x,y,w,b):
    m=x.shape[0]
    dj_dw=0
    dj_db=0
    
    for i in range(m):
        f_wb= w * x[i]+b
        dj_dw = dj_dw + (f_wb - y[i])*x[i]
        dj_db = dj_db + (f_wb - y[i])
    
    dj_dw = dj_dw/ m
    dj_db = dj_db /m
    
    return dj_dw,dj_db 

def gradient_descent(x,y,alpha,num_iter,gradient_compute):
    
    b=0
    w=0
    
    for i in range(num_iter):
        tmp_w,tmp_b=gradient_compute(x,y,w,b)
        w = w - alpha * tmp_w
        b = b - alpha * tmp_b
            
    return w,b


w_init=0
b_init=0
tmp_alpha=1e-2
iterations = 10000

w_final,b_final=gradient_descent(x_train,y_train,tmp_alpha,iterations,compute_gradient)

lin_model = LinearRegression()

lin_model.fit(x_train,y_train)

best_b = lin_model.intercept_
best_w = lin_model.coef_

pred1=simple_lin_model(2.7, best_w, best_b)
pred2=lin_model.predict([[2.7]])

print(pred1)
print(pred2)

x_=np.arange(min(x_train),max(x_train),1.5).reshape(-1,1)
y_head=lin_model.predict(x_)

plt.scatter(x,y,c='blue',label='Actuel Value')
plt.plot(x_,y_head,c='red',label='Predict Value')
plt.show()