#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 13:00:50 2024

@author: mali
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

np.random.seed(7)
x = np.random.uniform(0, 10, 100).reshape(-1, 1)  
y = np.sin(x).ravel() + np.random.normal(0, 0.1, x.shape[0])

data = pd.DataFrame({"Feature": x.ravel(),"Target": y})

plt.scatter(x,y,color='blue',label='data')
plt.xlabel('feature')
plt.ylabel('target')
plt.legend()
plt.show()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=7)


from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()

dt.fit(x_train,y_train)

y_head = dt.predict(x_test)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_head)

predict = dt.predict([x_train[0]])

print(predict)
print(y_train[0])
