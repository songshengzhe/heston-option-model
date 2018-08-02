# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 09:58:42 2018

@author: Administrator

"""
from generator import*
from Greek import*
from Heston import*
import numpy as np
import pandas as pd


def options_portfolio(filename,kappa,theta,sigma,rho,v0,r):
        
    data=pd.read_excel(filename,sheetname=0)
    n=len(data)
    data=data.reindex(columns=list(data.columns)+['Price','Delta','Gamma',
                      'Vega','Theta'])
    price=np.zeros(n)
    greeks=np.zeros((n,4))
    for i in range(n):
        if data['Type'][i]=='call':
            price[i]=Heston_call(kappa[0],theta[0],sigma[0],rho[0],v0[0],r,data['Time'][i],data['Spot'][i],data['Strike'][i])
            sample_path,variance,s=Heston_generator(kappa[0],theta[0],sigma[0],rho[0],v0[0],r,data['Time'][i],data['Spot'][i],num=10000,seed=None)
            greeks[i]=Heston_call_Greeks(sample_path,kappa[0],theta[0],sigma[0],rho[0],v0[0],r,data['Time'][i],data['Strike'][i],num=10000,seed=s,d=0.01)
        if data['Type'][i]=='put':
            price[i]=Heston_put(kappa[1],theta[1],sigma[1],rho[1],v0[1],r,data['Time'][i],data['Spot'][i],data['Strike'][i])
            sample_path,variance,s=Heston_generator(kappa[1],theta[1],sigma[1],rho[1],v0[1],r,data['Time'][i],data['Spot'][i],num=10000,seed=None)
            greeks[i]=Heston_put_Greeks(sample_path,kappa[1],theta[1],sigma[1],rho[1],v0[1],r,data['Time'][i],data['Strike'][i],num=10000,seed=s,d=0.01)
    
    data['Price']=price
    data[['Delta','Vega','Theta','Gamma']]=greeks
    data.loc[data['Position']=='long','Position']=1
    data.loc[data['Position']=='short','Position']=-1
    data_copy=data[['Price','Delta','Vega','Theta','Gamma']].apply(lambda x:x*data['Position'])
    data.loc['portfolio',['Price','Delta','Vega','Theta','Gamma']]=data_copy.apply(lambda x:x.sum())            
    data.to_csv('portfolio_result.csv')

def portfolio_price(data,kappa,theta,sigma,rho,v0,r):
    
    if type(kappa)==float:
        kappa=[kappa,kappa]
    if type(theta)==float:
        theta=[theta,theta]
    if type(sigma)==float:
        sigma=[sigma,sigma]
    if type(rho)==float:
        rho=[rho,rho]
    if type(v0)==float:
        v0=[v0,v0]
    
    
    n=len(data)
    data=data.reindex(columns=list(data.columns)+['Price','Delta','Gamma',
                      'Vega','Theta'])
    price=np.zeros(n)
    greeks=np.zeros((n,4))
    for i in range(n):
        if data['Type'][i]=='call':
            price[i]=Heston_call(kappa[0],theta[0],sigma[0],rho[0],v0[0],r,data['Time'][i],data['Spot'][i],data['Strike'][i])
            sample_path,variance,s=Heston_generator(kappa[0],theta[0],sigma[0],rho[0],v0[0],r,data['Time'][i],data['Spot'][i],num=10000,seed=None)
            greeks[i]=Heston_call_Greeks(sample_path,kappa[0],theta[0],sigma[0],rho[0],v0[0],r,data['Time'][i],data['Strike'][i],num=10000,seed=s,d=0.01)
        if data['Type'][i]=='put':
            price[i]=Heston_put(kappa[1],theta[1],sigma[1],rho[1],v0[1],r,data['Time'][i],data['Spot'][i],data['Strike'][i])
            sample_path,variance,s=Heston_generator(kappa[1],theta[1],sigma[1],rho[1],v0[1],r,data['Time'][i],data['Spot'][i],num=10000,seed=None)
            greeks[i]=Heston_put_Greeks(sample_path,kappa[1],theta[1],sigma[1],rho[1],v0[1],r,data['Time'][i],data['Strike'][i],num=10000,seed=s,d=0.01)
    
    data['Price']=price
    data[['Delta','Vega','Theta','Gamma']]=greeks
    data.loc[data['Position']=='long','Position']=1
    data.loc[data['Position']=='short','Position']=-1
    data_copy=data[['Price','Delta','Vega','Theta','Gamma']].apply(lambda x:x*data['Position'])
    data.loc['portfolio',['Price','Delta','Vega','Theta','Gamma']]=data_copy.apply(lambda x:x.sum())  
    
    return data.loc['portfolio',['Price','Delta','Vega','Theta','Gamma']]
    
def options_portfolio_change(filename,change_time,kappa,theta,sigma,rho,v0,r):
    change_result_price=pd.DataFrame(np.zeros((3,3)),index=[0.95,1.,1.05],columns=[0.95,1.,1.05])
    change_result_price.index.name='price&vol'
    
    change_result_delta=pd.DataFrame(np.zeros((3,3)),index=[0.95,1.,1.05],columns=[0.95,1.,1.05])
    change_result_delta.index.name='price&vol'
    
    change_result_gamma=pd.DataFrame(np.zeros((3,3)),index=[0.95,1.,1.05],columns=[0.95,1.,1.05])
    change_result_gamma.index.name='price&vol'
    
    
    change_result_vega=pd.DataFrame(np.zeros((3,3)),index=[0.95,1.,1.05],columns=[0.95,1.,1.05])
    change_result_vega.index.name='price&vol'
    
    change_result_theta=pd.DataFrame(np.zeros((3,3)),index=[0.95,1.,1.05],columns=[0.95,1.,1.05])
    change_result_theta.index.name='price&vol'
    

    
    data=pd.read_excel(filename,sheetname=0)
    
    for x in range(3):
        for y in range(3):
            data_copy=data.copy()
            data_copy['Spot']=change_result_price.index[x]*data['Spot']
            data_copy['Time']=data['Time']-change_time
            change_result=portfolio_price(data_copy,kappa,theta,sigma,rho,np.array(v0)*(change_result_price.columns[y]**2),r)
            change_result_price.iloc[x,y]=change_result['Price']
            change_result_delta.iloc[x,y]=change_result['Delta']
            change_result_gamma.iloc[x,y]=change_result['Gamma']
            change_result_vega.iloc[x,y]=change_result['Vega']
            change_result_theta.iloc[x,y]=change_result['Theta']
            del data_copy
            
    change_result_price.to_csv('change_result_price.csv')
    change_result_delta.to_csv('change_result_delta.csv')
    change_result_gamma.to_csv('change_result_gamma.csv')
    change_result_vega.to_csv('change_result_vega.csv')
    change_result_theta.to_csv('change_result_theta.csv')    
'''
r=0.03
sigma=0.6
rho=-0.04
theta=0.2
kappa=5.8
v0=0.04
if type(kappa)==float:
    kappa=[kappa,kappa]
if type(theta)==float:
    theta=[theta,theta]
if type(sigma)==float:
    sigma=[sigma,sigma]
if type(rho)==float:
    rho=[rho,rho]
if type(v0)==float:
    v0=[v0,v0]
filename='options_portfolio.xlsx'
options_portfolio('options_portfolio.xlsx',kappa,theta,sigma,rho,v0,r)


change_time=0.083

options_portfolio_change('options_portfolio.xlsx',change_time,kappa,theta,sigma,rho,v0,r)
'''