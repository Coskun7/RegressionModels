#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 19:15:51 2024

@author: mali
"""

import numpy as np

x_train = np.array([35,53,78,93,109,117,134,156,167,178,182,194,202,217,225])
y_train = np.array([300,553,619,787,859,1030,1186,1354,1453,1526,1611,1789,1832,1923,2002])

w_init = 10
b_init = 20

def compute_cost (x,y,w,b):
    
    m=x.shape[0]
    
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i]+ b
        
        cost = cost + (f_wb - y[i])**2
    
    total_cost = 1/(2*m)*cost
    
    return total_cost

# value of cost function is 21116.53

def compute_gradient(x,y,w,b):
    
    m = x.shape[0]
    
    dj_dw = 0
    
    dj_db = 0
    
    for i in range(m):
    
        f_wb = w * x[i] + b
        
        dj_dw = dj_dw + (f_wb-y[i])*x[i]
        
        dj_db = dj_db + (f_wb - y[i])
    
    dj_db = dj_db / m
    dj_dw = dj_dw / m
    
    return dj_db, dj_dw

derivative_b,derivative_w = compute_gradient(x_train,y_train,w_init,b_init)

#def com_gradient (x, y, w, b): 
    
#     m=x.shape[0]
#     dj_dw = 0
#     dj_db=0
    
#    for i in range(m):
#        f_wb = w * x[i] + b
#        dj_dw_i = (f_wb - y[i]) * x[i]
#        dj_db_i = f_wb - y[i]
#        dj_dw += dj_dw_i
#        dj_db += dj_db_i
#    dj_dw = dj_dw / m
#    dj_db = dj_db / m 
    
#    return dj_db,dj_dw THIS IS EXAMPLE I SEE IN COURSE I WANTED WRITE THİS TOO

def gradient_descent(x,y,w,b,alpha,num_iterator):
    
    
    for i in range(num_iterator):
        tmp_w,tmp_b= compute_gradient(x, y, w, b)
        w = w - alpha * tmp_w
        b = b - alpha * tmp_b

    
    return w,b

# I know this code can be improve because ı remembered this code is better at the course
# but ı can do this for now thank you for reading my code