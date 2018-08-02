# -*- coding: utf-8 -*-
"""
Created on Thu Jul 05 09:09:52 2018

@author: Administrator
"""

import numpy as np 
import scipy
from scipy.integrate import quad

j=complex(0,1)

def Hestf(phi,kappa,theta,sigma,rho,v0,r,T,s0,type):
    if type==1:
        u=0.5
        b=kappa-rho*sigma
    else:
        u=-0.5
        b=kappa
    
    a=kappa*theta
    x=np.log(s0)
    q=b-rho*sigma*phi*j
    d=np.sqrt(q**2-sigma**2*(2*u*phi*j-phi**2))
    g=(q+d)/(q-d)
    C=r*phi*j*T+a/sigma**2*((q+d)*T-2*np.log((1-g*np.exp(d*T))/(1-g)))
    D=(q+d)/sigma**2*((1-np.exp(d*T))/(1-g*np.exp(d*T)))
    
    f=np.exp(C+D*v0+j*phi*x)
    
    return f

    
    return f
def HestonPIntegrand(phi,kappa,theta,sigma,rho,v0,r,T,s0,K,type):
    ret=np.real(np.exp(-j*phi*np.log(K))*Hestf(phi,kappa,theta,sigma,rho,v0,r,T,s0,type)/(j*phi))
    return ret

def HestonP(kappa,theta,sigma,rho,v0,r,T,s0,K,type):
    ret=0.5+1./np.pi*quad(lambda x:HestonPIntegrand(x,kappa,theta,sigma,rho,v0,r,T,s0,K,type),0,100)[0]
    return ret

def Heston_call(kappa,theta,sigma,rho,v0,r,T,s0,K):
    return s0*HestonP(kappa,theta,sigma,rho,v0,r,T,s0,K,1)-K*np.exp(-r*T)*HestonP(kappa,theta,sigma,rho,v0,r,T,s0,K,2)

def Heston_put(kappa,theta,sigma,rho,v0,r,T,s0,K):
    return s0*HestonP(kappa,theta,sigma,rho,v0,r,T,s0,K,1)-K*np.exp(-r*T)*HestonP(kappa,theta,sigma,rho,v0,r,T,s0,K,2)-(s0-K*np.exp(-r*T))
'''
s0=100.
K=90.
T=1.
r=0.03
sigma=0.6
rho=-0.04
theta=0.2
kappa=5.8
v0=0.04

print Heston_call(kappa,theta,sigma,rho,v0,r,T,s0,K)
'''
#print HestonP(kappa,theta,sigma,rho,v0,r,T,s0,K,1)