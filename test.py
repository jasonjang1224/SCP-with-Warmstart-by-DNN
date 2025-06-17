# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 20:18:21 2025

@author: choi
"""
import os


os.chdir(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(os.path.abspath(__file__))
import matplotlib.pyplot as plt
#%matplotlib inline  
#%load_ext autoreload
#%autoreload 2
import numpy as np
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
#     print ("Values are: \n%s" % (x))

import sys
# sys.path.append('../')
#sys.path.append('/')
sys.path.append(os.path.join(current_dir, 'constraints'))
sys.path.append(os.path.join(current_dir, 'model'))
sys.path.append(os.path.join(current_dir, 'cost'))

import UnicycleModel

#%%
import UnicycleCost
import UnicycleConstraints
from scipy.integrate import solve_ivp
from PTR_tf_free_v2 import PTR_tf_free

start_time = time.time()

ix = 3
iu = 2
ih = 2
N = 30
tf = 3
delT = tf/N
max_iter = 20

#%%

mu=0
sigma=0.5

xi = np.zeros(3)
xi[0] = -1.0 +np.random.normal(mu,sigma)
xi[1] = -2.0 +np.random.normal(mu,sigma)
xi[2] = 0 #+np.random.normal(mu,sigma)

xf = np.zeros(3)
xf[0] = 2.0 +np.random.normal(mu,sigma)
xf[1] = 2.0 +np.random.normal(mu,sigma)
xf[2] = 0 #+np.random.normal(mu,sigma)

myModel = UnicycleModel.unicycle('Hello',ix,iu,'numeric_central')
myCost = UnicycleCost.unicycle('Hello',ix,iu,N)
myConst = UnicycleConstraints.UnicycleConstraints('Hello',ix,iu)

x0 = np.zeros((N+1,ix))
for i in range(N+1) :
    x0[i] = (N-i)/N * xi + i/N * xf
#u0 = np.random.rand(N+1,iu)
u0 = np.zeros((N+1,iu))
#u0 = np.ones((N+1, iu)) * np.pi * 50


#%%

i1 = PTR_tf_free('unicycle',N,tf,max_iter,myModel,myCost,myConst,type_discretization="zoh",
          w_c=1,w_vc=1e3,w_tr=1e-1,w_rate=0,
         tol_vc=1e-6,tol_tr=1e-3)
x,u,xbar,ubar,tf,total_num_iter,flag_boundary,l,l_vc,l_tr,x_traj,u_traj,T_traj = i1.run(x0,u0,xi,xf)

end_time = time.time()

print(f"실행 시간: {end_time - start_time:.4f}초")

#%%

t_index = np.array(range(N+1))*delT

plt.figure(figsize=(10,10))
fS = 18
plt.subplot(221)
plt.plot(x[:,0], x[:,1],'--', linewidth=2.0)
plt.plot(xbar[:,0], xbar[:,1],'-', linewidth=2.0)
plt.plot(xf[0],xf[1],"o",label='goal')
plt.gca().set_aspect('equal', adjustable='box')
plt.axis([-3, 3, -3, 3])
plt.xlabel('X (m)', fontsize = fS)
plt.ylabel('Y (m)', fontsize = fS)
plt.subplot(222)
plt.plot(t_index, xbar[:,0], linewidth=2.0,label='naive')
plt.xlabel('time (s)', fontsize = fS)
plt.ylabel('x1 (m)', fontsize = fS)
plt.subplot(223)
plt.plot(t_index, xbar[:,1], linewidth=2.0,label='naive')
plt.xlabel('time (s)', fontsize = fS)
plt.ylabel('x2 (m)', fontsize = fS)
plt.subplot(224)
plt.plot(t_index, xbar[:,2], linewidth=2.0,label='naive')
plt.xlabel('time (s)', fontsize = fS)
plt.ylabel('x3 (rad)', fontsize = fS)
plt.legend(fontsize=fS)
plt.show()

plt.figure()
plt.subplot(121)
if i1.type_discretization == "zoh" :
    plt.step(t_index, [*ubar[:N,0],ubar[N-1,0]],alpha=1.0,where='post',linewidth=2.0)
    plt.step(t_index, [*u[:N,0],u[N-1,0]],alpha=1.0,where='post',linewidth=2.0)
elif i1.type_discretization == "foh" :
    plt.plot(t_index, ubar[:,0], linewidth=2.0)
plt.xlabel('time (s)', fontsize = fS)
plt.ylabel('v (m/s)', fontsize = fS)
plt.subplot(122)
if i1.type_discretization == "zoh" :
    plt.step(t_index, [*ubar[:N,1],ubar[N-1,1]],alpha=1.0,where='post',linewidth=2.0)
elif i1.type_discretization == "foh" :
    plt.plot(t_index, ubar[:,1], linewidth=2.0)
plt.xlabel('time (s)', fontsize = fS)
plt.ylabel('w (rad/s)', fontsize = fS)
plt.show()

print(x)
print(tf)
