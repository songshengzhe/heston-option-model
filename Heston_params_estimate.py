# -*- coding: utf-8 -*-
"""
Created on Thu Jul 05 14:42:43 2018

@author: Administrator
"""

import Heston
import Greek
import generator
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

def options_data(filename,r):
    options=pd.read_excel(filename,sheetname=1)
    options['Mid']=(options['Bid']+options['Ask'])/2
    options['risk_free']=r
    return options

class Hestoncall_calibration:
    def __init__(self,call_data):
        self.call_data=call_data
        self.number_of_call=len(call_data)
        #记录data中的不同到期日数量
        self.spot_type=np.array(call_data['Spot'].unique())
        self.time_type=np.array(call_data['Time'].unique())
    
    def HestonDifference_call(self,x):
        price_difference=np.zeros(self.number_of_call)
        for i in range(self.number_of_call):
            price_difference[i]=(self.call_data['Mid'][i]-Heston.Heston_call((x[0]+x[2]**2)/(2*x[1]),x[1],x[2],x[3],x[4],self.call_data['risk_free'][i],self.call_data['Time'][i],self.call_data['Spot'][i],self.call_data['Strike'][i]))/np.sqrt(np.abs(self.call_data['Bid'][i]-self.call_data['Ask'][i]))
        return np.sum(price_difference**2)
        
    def call_calibration(self):
        x0=[5.,0.02,0.6,-0.4,0.04]
        bnds=((0,20),(0,1),(0,1),(-1,1),(0,1))
        res=minimize(self.HestonDifference_call,x0,method='SLSQP',bounds=bnds)
        
        x=res['x']
        kappa=(x[0]+x[2]**2)/(2*x[1])
        theta=x[1]
        sigma=x[2]
        rho=x[3]
        v0=x[4]
    
        return kappa,theta,sigma,rho,v0
        
    def generate_callmodel(self,kappa,theta,sigma,rho,v0):
        price_model=np.zeros(self.number_of_call)
        greeks_model=np.zeros((self.number_of_call,4))
        for i in range(self.number_of_call):
            price_model[i]=Heston.Heston_call(kappa,theta,sigma,rho,v0,self.call_data['risk_free'][i],self.call_data['Time'][i],self.call_data['Spot'][i],self.call_data['Strike'][i])
            sample_path,variance,s=generator.Heston_generator(kappa,theta,sigma,rho,v0,self.call_data['risk_free'][i],self.call_data['Time'][i],self.call_data['Spot'][i],num=10000,seed=None)
            greeks_model[i]=Greek.Heston_call_Greeks(sample_path,kappa,theta,sigma,rho,v0,self.call_data['risk_free'][i],self.call_data['Time'][i],self.call_data['Strike'][i],num=10000,seed=s,d=0.01)
        result=pd.DataFrame(columns=['Spot','Time','Strike','real','model','delta','gamma','vega','theta'])
        result[['Spot','Time','Strike','real']]=self.call_data[['Spot','Time','Strike','Mid']]
        result['model']=price_model
        result[['delta','vega','theta','gamma']]=greeks_model
        
        return result
        
    def compare_price(self,kappa,theta,sigma,rho,v0):
        real=np.array(self.call_data['Mid'])
        model=np.zeros(self.number_of_call)
        for i in range(self.number_of_call):
            model[i]=Heston.Heston_call(kappa,theta,sigma,rho,v0,self.call_data['risk_free'][i],self.call_data['Time'][i],self.call_data['Spot'][i],self.call_data['Strike'][i])
        for i in range(len(self.time_type)):
            index=np.array(self.call_data['Time']==self.time_type[i])
            strikes=self.call_data['Strike'][index]
            plt.figure(figsize=(7,5))
            plt.plot(np.array(strikes),np.array([real[index],model[index]]).T)
            plt.legend(['real','model'],loc=0)
            plt.xlabel('Strike')
            plt.ylabel('call Option price')
            plt.title('Spot=%.2f,T=%.2f'%(self.spot_type[i],self.time_type[i]))
            plt.show()
            plt.close()
            
class Hestonput_calibration:
    def __init__(self,put_data):
        self.put_data=put_data
        self.number_of_put=len(put_data)
        self.spot_type=np.array(put_data['Spot'].unique())
        self.time_type=np.array(put_data['Time'].unique())
        
    def HestonDifference_put(self,x):
        price_difference2=np.zeros(self.number_of_put)
        for i in range(self.number_of_put):
            price_difference2[i]=(self.put_data['Mid'][i]-Heston.Heston_put((x[0]+x[2]**2)/(2*x[1]),x[1],x[2],x[3],x[4],self.put_data['risk_free'][i],self.put_data['Time'][i],self.put_data['Spot'][i],self.put_data['Strike'][i]))/np.sqrt(np.abs(self.put_data['Bid'][i]-self.put_data['Ask'][i]))
        return np.sum(price_difference2**2)
        
    def put_calibration(self):
        
        x0=[5.,0.02,0.6,-0.4,0.04]
        bnds=((0,20),(0,1),(0,1),(-1,1),(0,1))
        res2=minimize(self.HestonDifference_put,x0,method='SLSQP',bounds=bnds)
        x2=res2['x']
        kappa2=(x2[0]+x2[2]**2)/(2*x2[1])
        theta2=x2[1]
        sigma2=x2[2]
        rho2=x2[3]
        v02=x2[4]
    
        return kappa2,theta2,sigma2,rho2,v02
        
    def generate_putmodel(self,kappa,theta,sigma,rho,v0):
        price_model=np.zeros(self.number_of_put)
        greeks_model=np.zeros((self.number_of_put,4))
        for i in range(self.number_of_put):
            price_model[i]=Heston.Heston_put(kappa,theta,sigma,rho,v0,self.put_data['risk_free'][i],self.put_data['Time'][i],self.put_data['Spot'][i],self.put_data['Strike'][i])
            sample_path,variance,s=generator.Heston_generator(kappa,theta,sigma,rho,v0,self.put_data['risk_free'][i],self.put_data['Time'][i],self.put_data['Spot'][i],num=10000,seed=None)
            greeks_model[i]=Greek.Heston_put_Greeks(sample_path,kappa,theta,sigma,rho,v0,self.put_data['risk_free'][i],self.put_data['Time'][i],self.put_data['Strike'][i],num=10000,seed=s,d=0.01)
        result=pd.DataFrame(columns=['Spot','Time','Strike','real','model','delta','gamma','vega','theta'])
        result[['Spot','Time','Strike','real']]=self.put_data[['Spot','Time','Strike','Mid']]
        result['model']=price_model
        result[['delta','vega','theta','gamma']]=greeks_model
        
        return result
    
    def compare_price(self,kappa,theta,sigma,rho,v0):
        real=np.array(self.put_data['Mid'])
        model=np.zeros(self.number_of_put)
        for i in range(self.number_of_put):
            model[i]=Heston.Heston_put(kappa,theta,sigma,rho,v0,self.put_data['risk_free'][i],self.put_data['Time'][i],self.put_data['Spot'][i],self.put_data['Strike'][i])
        for i in range(len(self.time_type)):
            index=np.array(self.put_data['Time']==self.time_type[i])
            strikes=self.put_data['Strike'][index]
            plt.figure(figsize=(7,5))
            plt.plot(np.array(strikes),np.array([real[index],model[index]]).T)
            plt.legend(['real','model'],loc=0)
            plt.xlabel('Strike')
            plt.ylabel('put Option price')
            plt.title('Spot=%.2f,T=%.2f'%(self.spot_type[i],self.time_type[i]))
            plt.show()
            plt.close()

'''
call_data=options_data('call20180710.xlsx')
put_data=options_data('put20180710.xlsx')

time_start=time.time()
x1=Hestoncall_calibration(call_data)
kappa,theta,sigma,rho,v0=169.79,0.0567,0.9324,-0.7955,0.0847
time_end=time.time()
print('totally cost',time_end-time_start)

#result1=x1.generate_callmodel(kappa,theta,sigma,rho,v0)

'''
'''    
time_start=time.time()
x2=Hestonput_calibration(put_data)
kappa2,theta2,sigma2,rho2,v02=x2.put_calibration()
time_end=time.time()
print('totally cost',time_end-time_start)

result2=x2.generate_putmodel(kappa,theta,sigma,rho,v0)
'''
'''
time_start=time.time()
kappa2,theta2,sigma2,rho2,v02=put_calibration()
time_end=time.time()
print('totally cost',time_end-time_start)

result2=generate_putmodel(kappa2,theta2,sigma2,rho2,v02)


generator.Heston_variance_plot(kappa,theta,sigma,rho,v0)
generator.Heston_variance_plot(kappa2,theta2,sigma2,rho2,v02)
'''