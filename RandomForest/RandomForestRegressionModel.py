#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 12:33:41 2024

@author: mali
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

# Rastgele veri seti boyutları
num_samples = 10000  # Örnek sayısı
num_features = 10    # Özellik sayısı

# Sayısal veriler için rastgele değerler oluştur
X, y = make_regression(n_samples=num_samples,n_features=num_features,noise=0.5,random_state=7)



# Kategorik veriler oluşturma
np.random.seed(7)
cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
jobs = ['Engineer', 'Teacher', 'Doctor', 'Artist', 'Lawyer']

data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(1, num_features + 1)])
data['city'] = np.random.choice(cities, size=num_samples)
data['job'] = np.random.choice(jobs, size=num_samples)
data['experience_years'] = np.random.randint(0, 30, size=num_samples)
data['education_level'] = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], size=num_samples)

# Hedef değişkeni ekleme
data['target'] = y + np.random.rand(num_samples) * 10  # Gürültü eklenmiş hedef

# İlk birkaç satırı göster
print(data.head())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = pd.get_dummies(data,columns=['city','job','education_level'])  


x = data.iloc[:,:14].to_numpy()
y = data.iloc[:,14].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=7)

RF = RandomForestRegressor(n_estimators=200,random_state=7)

RF.fit(x_train, y_train)

y_pred = RF.predict(x_test)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(mse) 

print(data['target'].describe())

from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse}")



