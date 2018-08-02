# -*- coding: utf-8 -*-
"""
Created on Mon Jul 09 13:44:20 2018

@author: Administrator
"""

from Tkinter import *
from Greek import * 
from Heston import *
from Heston_params_estimate import *
from generator import*
from options_portfolio import *
import tkMessageBox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2TkAgg
import scipy.special._ufuncs_cxx
import scipy.linalg.cython_lapack

class Application:   
    def __init__(self):
        self.root=Tk()
        self.root.title('Heston model pricing')
        self.root.geometry('800x800')
        
        self.r=StringVar()
        self.r.set('0.03')

        self.timestart=0
        self.timeend=0
        self.kappa=0
        self.theta=0
        self.sigma=0
        self.rho=0
        self.v0=0
        self.kappa2=0
        self.theta2=0
        self.sigma2=0
        self.rho2=0
        self.v02=0
        
        
        self.entry_kappa=StringVar()
        self.entry_theta=StringVar()
        self.entry_sigma=StringVar()
        self.entry_rho=StringVar()
        self.entry_v0=StringVar()
        self.entry_type=StringVar()
        
        self.entry_kappa.set('5.8')
        self.entry_theta.set('0.05')
        self.entry_sigma.set('0.6')
        self.entry_rho.set('-0.04')
        self.entry_v0.set('0.04')
        self.entry_type.set('call')
        
        self.spot=StringVar()
        self.strike=StringVar()
        self.time=StringVar()
        
        self.spot.set('100')
        self.strike.set('100')
        self.time.set('0.083')
        
        self.x=Hestoncall_calibration(pd.DataFrame(columns=['Spot','Time','Strike','real','model','delta','gamma','vega','theta','Mid']))
        self.x2=Hestonput_calibration(pd.DataFrame(columns=['Spot','Time','Strike','real','model','delta','gamma','vega','theta','Mid']))
        
    
        self.frm=LabelFrame(self.root,width=780,height=200,text='calibration')
        self.frm.grid(row=0,column=0,padx=15)
        #params calibration part
        Button(self.frm,text='call option data file path(Button to read)',command=self.get_call).place(x=10,y=15)
        self.entry1=Entry(self.frm)
        self.entry1.place(x=270,y=15)
        Button(self.frm,text='call calibration',command=self.call_calibration).place(x=450,y=15)
        Button(self.frm,text='show call params',command=self.showcallparams).place(x=600,y=15)
        
        
        Button(self.frm,text='put option data file path(Button to read)',command=self.get_put).place(x=10,y=65)
    
        self.entry2=Entry(self.frm)
        self.entry2.place(x=270,y=65)
        Button(self.frm,text='put calibration',command=self.put_calibration).place(x=450,y=65)
        Button(self.frm,text='show put params',command=self.showputparams).place(x=600,y=65)        
        
        Label(self.frm,text='set risk_free').place(x=10,y=115)
        Entry(self.frm,textvariable=self.r).place(x=100,y=115)
        
        self.frm2=LabelFrame(self.root,width=780,height=580,text='Entry params you want')
        self.frm2.grid(row=1,column=0,padx=15)      
        
        Label(self.frm2,text='kappa').place(x=10,y=15)
        Entry(self.frm2,textvariable=self.entry_kappa).place(x=60,y=15)

        Label(self.frm2,text='theta').place(x=210,y=15)
        Entry(self.frm2,textvariable=self.entry_theta).place(x=260,y=15)
        
        Label(self.frm2,text='sigma').place(x=410,y=15)
        Entry(self.frm2,textvariable=self.entry_sigma).place(x=460,y=15)


        Label(self.frm2,text='rho').place(x=10,y=65)
        Entry(self.frm2,text=self.entry_rho).place(x=60,y=65)

        Label(self.frm2,text='v0').place(x=210,y=65)
        Entry(self.frm2,textvariable=self.entry_v0).place(x=260,y=65)
        
        Label(self.frm2,text='option type').place(x=410,y=65)
        Entry(self.frm2,textvariable=self.entry_type).place(x=500,y=65)
        
        Button(self.frm2,text='MC use calibration params',command=self.generate_calibration_report).place(x=40,y=110)
        Button(self.frm2,text='MC use entry params',command=self.generate_entry_report).place(x=400,y=110)
        
        Button(self.frm2,text='MC plot vol and log_return use calibration params',command=self.calibration_volatility_logreturn_show).place(x=40,y=160)
        Button(self.frm2,text='MC plot vol and log_return use entry params',command=self.entry_volatility_logreturn_show).place(x=400,y=160)
        
        Label(self.frm2,text='Spot').place(x=10,y=220)
        Entry(self.frm2,textvariable=self.spot).place(x=60,y=220)
        Label(self.frm2,text='Strike').place(x=210,y=220)
        Entry(self.frm2,textvariable=self.strike).place(x=260,y=220)
        Label(self.frm2,text='Time').place(x=410,y=220)
        Entry(self.frm2,textvariable=self.time).place(x=460,y=220)
        
        Button(self.frm2,text='call(calibration params)',command=self.call_calculate_calibration).place(x=40,y=270)
        Button(self.frm2,text='compare',command=self.call_compare_calibration).place(x=240,y=270)
        Button(self.frm2,text='put(calibration params)',command=self.put_calculate_calibration).place(x=400,y=270)
        Button(self.frm2,text='compare',command=self.put_compare_calibration).place(x=600,y=270)
        
        Button(self.frm2,text='call(entry params)',command=self.call_caculate_entry).place(x=40,y=320)
        Button(self.frm2,text='compare',command=self.call_compare_entry).place(x=240,y=320)
        Button(self.frm2,text='put(entry params)',command=self.put_caculate_entry).place(x=400,y=320)
        Button(self.frm2,text='compare',command=self.put_compare_entry).place(x=600,y=320)
    
        Label(self.frm2,text='portfolio_data').place(x=10,y=400)
        self.Entry3=Entry(self.frm2)
        self.Entry3.place(x=100,y=400)
        Button(self.frm2,text='calibration params',command=self.portfolio_report_calibration).place(x=400,y=400)
        Button(self.frm2,text='entry params',command=self.portfolio_report_entry).place(x=600,y=400)
        
        Label(self.frm2,text='time change').place(x=10,y=460)
        self.change_time=StringVar()
        self.change_time.set('0.083')
        Entry(self.frm2,textvariable=self.change_time).place(x=100,y=460)
        Button(self.frm2,text='show time change report',command=self.timechange_portfoli_result).place(x=400,y=460)
        
        
        
    def get_call(self):
        filename=self.entry1.get()
        r=float(self.r.get())
        self.call_data=options_data(filename,r)
        self.x=Hestoncall_calibration(self.call_data)

    def get_put(self):
        filename=self.entry2.get()
        r=float(self.r.get())
        self.put_data=options_data(filename,r)
        self.x2=Hestonput_calibration(self.put_data)
        
    def call_calibration(self):
        filename=self.entry1.get()
        r=float(self.r.get())
        self.call_data=options_data(filename,r)
        self.timestart=time.time()
        self.x=Hestoncall_calibration(self.call_data)
        self.kappa,self.theta,self.sigma,self.rho,self.v0=self.x.call_calibration()
        self.timeend=time.time()
        tkMessageBox.showinfo(title='running cost time',message=self.timeend-self.timestart)
        
    def put_calibration(self):
        filename=self.entry2.get()
        r=float(self.r.get())
        self.put_data=options_data(filename,r)
        self.timestart=time.time()
        self.x2=Hestonput_calibration(self.put_data)
        self.kappa2,self.theta2,self.sigma2,self.rho2,self.v02=self.x2.put_calibration()
        self.timeend=time.time()
        tkMessageBox.showinfo(title='running cost time',message=self.timeend-self.timestart)
    
    def showcallparams(self):
        tkMessageBox.showinfo(title='call params',message="Kappa:%.4f\n Theta:%.4f\n Sigma:%.4f\n rho:%.4f\n v0:%.4f"%(self.kappa,self.theta,self.sigma,self.rho,self.v0))

    def showputparams(self):
        tkMessageBox.showinfo(title='call params',message="Kappa:%.4f\n Theta:%.4f\n Sigma:%.4f\n rho:%.4f\n v0:%.4f"%(self.kappa2,self.theta2,self.sigma2,self.rho2,self.v02))
        
    def generate_calibration_report(self):
        if self.entry_type.get()=='call':
            kappa=self.kappa
            theta=self.theta
            sigma=self.sigma
            rho=self.rho
            v0=self.v0
            result=(self.x).generate_callmodel(kappa,theta,sigma,rho,v0)
            result.to_csv("call_model_result.csv")
        if self.entry_type.get()=='put':
            kappa=self.kappa2
            theta=self.theta2
            sigma=self.sigma2
            rho=self.rho2
            v0=self.v02
            result2=(self.x2).generate_putmodel(kappa,theta,sigma,rho,v0)
            result2.to_csv("put_model_result.csv")
     
    def generate_entry_report(self):
        if self.entry_type.get()=='call':
            kappa=float(self.entry_kappa.get())
            theta=float(self.entry_theta.get())
            sigma=float(self.entry_sigma.get())
            rho=float(self.entry_rho.get())
            v0=float(self.entry_v0.get())
            
            result=(self.x).generate_callmodel(kappa,theta,sigma,rho,v0)
            result.to_csv("call_model_result.csv")
            
        if self.entry_type.get()=='put':
            kappa=float(self.entry_kappa.get())
            theta=float(self.entry_theta.get())
            sigma=float(self.entry_sigma.get())
            rho=float(self.entry_rho.get())
            v0=float(self.entry_v0.get())
            
            result=(self.x2).generate_putmodel(kappa,theta,sigma,rho,v0)
            result.to_csv("put_model_result.csv")
    
    def calibration_volatility_logreturn_show(self):
        if self.entry_type.get()=='call':
            picture1,picture2=Heston_variance_plot(self.kappa,self.theta,self.sigma,self.rho,self.v0)
            kappa=self.kappa
            theta=self.theta
            sigma=self.sigma
            rho=self.rho
            v0=self.v0
        if self.entry_type.get()=='put':
            picture1,picture2=Heston_variance_plot(self.kappa2,self.theta2,self.sigma2,self.rho2,self.v02)
            kappa=self.kappa2
            theta=self.theta2
            sigma=self.sigma2
            rho=self.rho2
            v0=self.v02       
        
        tk=Toplevel()
        tk.geometry('800x800')
        canvas=FigureCanvasTkAgg(picture1,master=tk)
        canvas.show()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        canvas=FigureCanvasTkAgg(picture2,master=tk)
        canvas.show()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        Label(tk,text="Kappa:%.4f Theta:%.4f Sigma:%.4f rho:%.4f v0:%.4f"%(kappa,theta,sigma,rho,v0)).pack(side=TOP, fill=BOTH, expand=1)
            
        toolbar =NavigationToolbar2TkAgg(canvas,tk)
        toolbar.update()
        canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)
        
        
    def entry_volatility_logreturn_show(self):
        
        kappa=float(self.entry_kappa.get())
        theta=float(self.entry_theta.get())
        sigma=float(self.entry_sigma.get())
        rho=float(self.entry_rho.get())
        v0=float(self.entry_v0.get())
        
        picture1,picture2=Heston_variance_plot(kappa,theta,sigma,rho,v0)
        
        tk=Toplevel()
        tk.geometry('800x800')
        canvas=FigureCanvasTkAgg(picture1,master=tk)
        canvas.show()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        canvas=FigureCanvasTkAgg(picture2,master=tk)
        canvas.show()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        
        Label(tk,text="Kappa:%.4f Theta:%.4f Sigma:%.4f rho:%.4f v0:%.4f"%(kappa,theta,sigma,rho,v0)).pack(side=TOP, fill=BOTH, expand=1)
            
        toolbar =NavigationToolbar2TkAgg(canvas,tk)
        toolbar.update()
        canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)
        
        
    def call_calculate_calibration(self):
        kappa=self.kappa
        theta=self.theta
        sigma=self.sigma
        rho=self.rho
        v0=self.v0
        r=float(self.r.get())
        T=float(self.time.get())
        s0=float(self.spot.get())
        K=float(self.strike.get())
        sample_path,variance,s=Heston_generator(kappa,theta,sigma,rho,v0,r,T,s0,num=10000,seed=None)
        price=Heston_call(kappa,theta,sigma,rho,v0,r,T,s0,K)
        [delta,vega,theta,gamma]=Heston_call_Greeks(sample_path,kappa,theta,sigma,rho,v0,r,T,K,num=10000,seed=s,d=0.01)
        tkMessageBox.showinfo(title='call',message='price:%.4f\n\n Delta:%.4f\n Gamma:%.4f\n Vega:%.4f\n Theta:%.4f'%(price,delta,gamma,vega,theta))
    
    def put_calculate_calibration(self):
        kappa=self.kappa2
        theta=self.theta2
        sigma=self.sigma2
        rho=self.rho2
        v0=self.v02
        r=float(self.r.get())
        T=float(self.time.get())
        s0=float(self.spot.get())
        K=float(self.strike.get())
        sample_path,variance,s=Heston_generator(kappa,theta,sigma,rho,v0,r,T,s0,num=10000,seed=None)
        price=Heston_put(kappa,theta,sigma,rho,v0,r,T,s0,K)
        [delta,vega,theta,gamma]=Heston_put_Greeks(sample_path,kappa,theta,sigma,rho,v0,r,T,K,num=10000,seed=s,d=0.01)
        tkMessageBox.showinfo(title='put',message='price:%.4f\n\n Delta:%.4f\n Gamma:%.4f\n Vega:%.4f\n Theta:%.4f'%(price,delta,gamma,vega,theta))        
    def call_caculate_entry(self):
        kappa=float(self.entry_kappa.get())
        theta=float(self.entry_theta.get())
        sigma=float(self.entry_sigma.get())
        rho=float(self.entry_rho.get())
        v0=float(self.entry_v0.get())
        r=float(self.r.get())
        T=float(self.time.get())
        s0=float(self.spot.get())
        K=float(self.strike.get())
        sample_path,variance,s=Heston_generator(kappa,theta,sigma,rho,v0,r,T,s0,num=10000,seed=None)
        price=Heston_call(kappa,theta,sigma,rho,v0,r,T,s0,K)
        [delta,vega,theta,gamma]=Heston_call_Greeks(sample_path,kappa,theta,sigma,rho,v0,r,T,K,num=10000,seed=s,d=0.01)
        tkMessageBox.showinfo(title='call',message='price:%.4f\n\n Delta:%.4f\n Gamma:%.4f\n Vega:%.4f\n Theta:%.4f'%(price,delta,gamma,vega,theta))
        
    def put_caculate_entry(self):
        kappa=float(self.entry_kappa.get())
        theta=float(self.entry_theta.get())
        sigma=float(self.entry_sigma.get())
        rho=float(self.entry_rho.get())
        v0=float(self.entry_v0.get())
        r=float(self.r.get())
        T=float(self.time.get())
        s0=float(self.spot.get())
        K=float(self.strike.get())
        sample_path,variance,s=Heston_generator(kappa,theta,sigma,rho,v0,r,T,s0,num=10000,seed=None)
        price=Heston_put(kappa,theta,sigma,rho,v0,r,T,s0,K)
        [delta,vega,theta,gamma]=Heston_put_Greeks(sample_path,kappa,theta,sigma,rho,v0,r,T,K,num=10000,seed=s,d=0.01)
        tkMessageBox.showinfo(title='put',message='price:%.4f\n\n Delta:%.4f\n Gamma:%.4f\n Vega:%.4f\n Theta:%.4f'%(price,delta,gamma,vega,theta))
    
    def call_compare_calibration(self):
        kappa=self.kappa
        theta=self.theta
        sigma=self.sigma
        rho=self.rho
        v0=self.v0
        self.x.compare_price(kappa,theta,sigma,rho,v0)
    def put_compare_calibration(self):
        kappa=self.kappa2
        theta=self.theta2
        sigma=self.sigma2
        rho=self.rho2
        v0=self.v02
        self.x2.compare_price(kappa,theta,sigma,rho,v0)
        
    def call_compare_entry(self):
        kappa=float(self.entry_kappa.get())
        theta=float(self.entry_theta.get())
        sigma=float(self.entry_sigma.get())
        rho=float(self.entry_rho.get())
        v0=float(self.entry_v0.get())
        self.x.compare_price(kappa,theta,sigma,rho,v0)
        
    def put_compare_entry(self):
        kappa=float(self.entry_kappa.get())
        theta=float(self.entry_theta.get())
        sigma=float(self.entry_sigma.get())
        rho=float(self.entry_rho.get())
        v0=float(self.entry_v0.get())
        self.x2.compare_price(kappa,theta,sigma,rho,v0)
        
    def portfolio_report_calibration(self):
        filename=self.Entry3.get()
        kappa=[self.kappa,self.kappa2]
        theta=[self.theta,self.theta2]
        sigma=[self.sigma,self.sigma2]
        rho=[self.rho,self.rho2]
        v0=[self.v0,self.v02]
        r=float(self.r.get())
        options_portfolio(filename,kappa,theta,sigma,rho,v0,r)
        
    def portfolio_report_entry(self):
        filename=self.Entry3.get()
        kappa=[self.entry_kappa,self.entry_kappa]
        theta=[self.entry_theta,self.entry_theta]
        sigma=[self.entry_sigma,self.entry_sigma]
        rho=[self.entry_rho,self.entry_rho]
        v0=[self.entry_rho,self.entry_rho]
        r=float(self.r.get())
        options_portfolio(filename,kappa,theta,sigma,rho,v0,r)
        
    def timechange_portfoli_result(self):
        filename=self.Entry3.get()
        kappa=[self.kappa,self.kappa2]
        theta=[self.theta,self.theta2]
        sigma=[self.sigma,self.sigma2]
        rho=[self.rho,self.rho2]
        v0=[self.v0,self.v02]
        r=float(self.r.get())
        change_time=float(self.change_time.get())
        options_portfolio_change(filename,change_time,kappa,theta,sigma,rho,v0,r)
            
             
if __name__=='__main__':
    app=Application()
    mainloop()