# -*- coding: utf-8 -*-
"""
Created on Wed Jul 04 10:20:47 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import *
import time

def BS_generator(sigma,S0,t,r=0.03,num=10000,seed=None):
    n=100.
    dt=float(t)/n
    sample_path=np.zeros((n+1,num))
    sample_path[0]=S0*np.ones(num)
    
    if seed==None:
        s=np.random.choice(np.arange(1,2000))
    else:
        s=seed
        
    np.random.seed(s)    
    rand=np.random.randn(n,num)
    
    for i in np.arange(1,n+1):
        sample_path[i]=sample_path[i-1]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*rand[i-1])
    
    return sample_path,s

def Heston_generator(kappa,theta,sigma,rho,v0,r,T,s0,num=10000,seed=None):
    n=100
    dt=float(T)/n
    
    sample_path=np.zeros((n+1,num))
    sample_path[0]=s0*np.ones(num)
    
    variance=np.zeros((n+1,num))
    variance[0]=v0*np.ones(num)
    
    if seed==None:
        s=np.random.choice(np.arange(1,2000))
    else:
        s=seed
    
    np.random.seed(s)
    rand_W=np.random.randn(n,num)
    rand_Z=np.random.randn(n,num)
    
    for i in np.arange(1,n+1):
        variance[i]=variance[i-1]+kappa*(theta-variance[i-1])*dt+sigma*np.sqrt(variance[i-1])*np.sqrt(dt)*rand_W[i-1]
        variance[i]=np.abs(variance[i])
        sample_path[i]=sample_path[i-1]+r*sample_path[i-1]*dt+np.sqrt(variance[i-1])*sample_path[i-1]*(
            rho*np.sqrt(dt)*rand_W[i-1]+np.sqrt(1-rho**2)*np.sqrt(dt)*rand_Z[i-1])        
    
    return sample_path,variance,s

def Heston_variance_plot(kappa,theta,sigma,rho,v0,r=0.03,T=1,s0=100,num=10000,seed=None):
    sample_path,variance,s=Heston_generator(kappa,theta,sigma,rho,v0,r,T,s0,num=10000,seed=None)
    
    picture1=plt.figure(figsize=(6,4))
    sns.distplot(np.sqrt(variance[-1]),ax=picture1.gca(),norm_hist=True,color='g',fit=norm,label='vol')
    plt.xlabel('volatility')
    plt.ylabel('density')
    plt.legend(loc=0)
    plt.title('volatility distrubtion')
    plt.close()
    
    picture2=plt.figure(figsize=(6,4))
    sns.distplot(np.log(sample_path[-1]/sample_path[0]),ax=picture2.gca(),norm_hist=True,color='g',fit=norm,label='log_r')
    plt.xlabel('return(default T=1)')
    plt.ylabel('density')
    plt.legend(loc=0)
    plt.title('return(default T=1) distrubtion')
    plt.close()
    
    return picture1,picture2