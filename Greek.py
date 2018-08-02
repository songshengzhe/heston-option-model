# -*- coding: utf-8 -*-
"""
Created on Mon Jul 02 17:43:47 2018

@author: Administrator
"""
import numpy as np
import scipy.stats as sts
from generator import BS_generator
from generator import Heston_generator


def MC_call_price(sample_path,t,K,r=0.03):
    price=np.exp(-r*t)*np.mean((sample_path[-1]-K)*((sample_path[-1]-K)>0))
    return price
    
def MC_put_price(sample_path,t,K,r=0.03):
    price=np.exp(-r*t)*np.mean((K-sample_path[-1])*((sample_path[-1]-K)<0))
    return price
    
def BS_call_Greeks(sample_path,K,sigma,t,r=0.03,seed=None,d=0.01):
    S0=sample_path[0,0]    
    
    sample_path1=(1+d)*sample_path
    sample_path2=(1-d)*sample_path
    price0=MC_call_price(sample_path,t,K,r)
    price1=MC_call_price(sample_path1,t,K,r)
    price2=MC_call_price(sample_path2,t,K,r)
    
    delta=(price1-price2)/(2*d*S0)
    gamma=(price2+price1-2*price0)/(d*S0)**2
    
    
    sample_path3=BS_generator((1+d)*sigma,S0,t,r,num=10000,seed=seed)[0]
    price3=MC_call_price(sample_path3,t,K,r)
    vega=(price3-price0)/(d*sigma)/100
    
    sample_path4=BS_generator(sigma,S0,(1+d)*t,r,num=10000,seed=seed)[0]
    price4=MC_call_price(sample_path4,(1+d)*t,K,r)
    theta=-(price4-price0)/(d*t)/365
    
    return [delta,vega,theta,gamma]
    
def Heston_call_Greeks(sample_path,kappa,theta,sigma,rho,v0,r,T,K,num=10000,seed=None,d=0.01):
    s0=sample_path[0,0]    
    
    sample_path1=Heston_generator(kappa,theta,sigma,rho,v0,r,T,(1+d)*s0,num=10000,seed=seed)[0]
    sample_path2=Heston_generator(kappa,theta,sigma,rho,v0,r,T,(1-d)*s0,num=10000,seed=seed)[0]
    price0=MC_call_price(sample_path,T,K,r)
    price1=MC_call_price(sample_path1,T,K,r)
    price2=MC_call_price(sample_path2,T,K,r)
    
    delta=(price1-price2)/(2*d*s0)
    gamma=(price2+price1-2*price0)/(d*s0)**2
    
    
    sample_path3=Heston_generator(kappa,theta,sigma,rho,(1+d)**2*v0,r,T,s0,num=10000,seed=seed)[0]
    price3=MC_call_price(sample_path3,T,K,r)
    vega=(price3-price0)/(d*np.sqrt(v0))/100
        
    sample_path4=Heston_generator(kappa,theta,sigma,rho,v0,r,(1+d)*T,s0,num=10000,seed=seed)[0]
    price4=MC_call_price(sample_path4,(1+d)*T,K,r)
    theta=-(price4-price0)/(d*T)/365
    
    del sample_path1
    del sample_path2
    del sample_path3
    del sample_path4
    
    return [delta,vega,theta,gamma]
    
def Heston_put_Greeks(sample_path,kappa,theta,sigma,rho,v0,r,T,K,num=10000,seed=None,d=0.01):
    s0=sample_path[0,0]    
    
    sample_path1=Heston_generator(kappa,theta,sigma,rho,v0,r,T,(1+d)*s0,num=10000,seed=seed)[0]
    sample_path2=Heston_generator(kappa,theta,sigma,rho,v0,r,T,(1-d)*s0,num=10000,seed=seed)[0]
    price0=MC_put_price(sample_path,T,K,r)
    price1=MC_put_price(sample_path1,T,K,r)
    price2=MC_put_price(sample_path2,T,K,r)
    
    delta=(price1-price2)/(2*d*s0)
    gamma=(price2+price1-2*price0)/(d*s0)**2
    
    
    sample_path3=Heston_generator(kappa,theta,sigma,rho,(1+d)**2*v0,r,T,s0,num=10000,seed=seed)[0]
    price3=MC_put_price(sample_path3,T,K,r)
    vega=(price3-price0)/(d*np.sqrt(v0))/100
        
    sample_path4=Heston_generator(kappa,theta,sigma,rho,v0,r,(1+d)*T,s0,num=10000,seed=seed)[0]
    price4=MC_put_price(sample_path4,(1+d)*T,K,r)
    theta=-(price4-price0)/(d*T)/365
    
    del sample_path1
    del sample_path2
    del sample_path3
    del sample_path4
    
    return [delta,vega,theta,gamma]
def Formula_call_price(S0,r,t,K,sigma):
    d1=(np.log(S0/K)+(r+0.5*(sigma**2))*t)/(sigma*np.sqrt(t))
    d2=(np.log(S0/K)+(r-0.5*(sigma**2))*t)/(sigma*np.sqrt(t))
    price=S0*sts.norm.cdf(d1)-K*np.exp(-r*t)*sts.norm.cdf(d2)
    return price

def Formula_call_Greeks(S0,r,t,K,sigma):
    d1=(np.log(S0/K)+(r+0.5*(sigma**2))*t)/(sigma*np.sqrt(t))
    d2=(np.log(S0/K)+(r-0.5*(sigma**2))*t)/(sigma*np.sqrt(t))
    delta=sts.norm.cdf(d1)
    vega=S0*sts.norm.pdf(d1)*np.sqrt(t)
    theta=-S0*sts.norm.pdf(d1)*sigma/(2*np.sqrt(t))-r*K*np.exp(-r*t)*sts.norm.cdf(d2)
    gamma=sts.norm.pdf(d1)/(S0*sigma*np.sqrt(t))
    vega=vega/100.
    theta=theta/365.
    return [delta,vega,theta,gamma]
    

    
