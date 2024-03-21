import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
plt.close('all')

# 3.0 -> u(t) = t^2*e^{-2t}
def dydt(x,t):
  dydt=[0,0,0]
  dydt[0]=x[1]
  dydt[1]=x[2]
  dydt[2]=t**2*np.exp(-2*t)-16*x[2]-79*x[1]-120*x[0]
  return dydt

y0=[0,0,0]

def an(t): #wyznaczone rozwiązanie analityczne
    return (1/18*np.exp(-2*t)*(t**2) - 1/6*np.exp(-2*t)*t + 61/324*np.exp(-2*t) -
             1/5*np.exp(-3*t) + 1/81*np.exp(-5*t) - 1/1620*np.exp(-8*t))

# 4.0 -> u(t) = delta diraca & u(t) = delta kroneckera
def dydt_k(x,t):
    dydt=[0,0,0]
    dydt[0]=x[1]
    dydt[1]=x[2]
    dydt[2]=1-16*x[2]-79*x[1]-120*x[0]
    return dydt

def an2(t):
    return (1/120)+(1/30)*np.exp(-5*t)-(1/120)*np.exp(-8*t)-(1/30)*np.exp(-3*t)

def an3(t):
    return -(1/6)*np.exp(-5*t)+(1/15)*np.exp(-8*t)+(1/10)*np.exp(-3*t)


t=np.linspace(0,8,100)
y3=odeint(dydt,y0,t)
y4=odeint(dydt_k, y0, t)
y5=np.gradient(y4[:,0], t)

y0_1=[0.1,0,0]
y0_2=[0,0.1,0]
y0_3=[0,0,0.1]

y3_1=odeint(dydt,y0_1,t)
y4_1=odeint(dydt_k, y0_1, t)

figure, axis = plt.subplots(3)
axis[0].plot(t, y3_1[:, 0],linewidth=3,label="solver", color='blue')
axis[0].plot(t, an(t),'--',linewidth=2,label="analityczne", color='red')
axis[1].plot(t,y4_1[:,0],linewidth=3,label="solver", color='blue')
axis[1].plot(t,an2(t),'--',linewidth=2,label="analityczne", color='red')
axis[2].plot(t,y5,linewidth=3,label="solver", color='blue')
axis[2].plot(t,an3(t),'--',linewidth=2,label="analityczne", color='red')
axis[0].grid()
axis[0].legend()
axis[1].grid()
axis[1].legend()
axis[2].grid()
axis[2].legend()
plt.show()
