# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 20:18:21 2025

@author: choi
"""
import os
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
from PTR import PTR

start_time = time.time()

class InitialToControlMLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, output_dim=62, dropout_prob=0.4):
        super(InitialToControlMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),  

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),  

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
    
checkpoint = torch.load(('control_mlp_v_tf_free.pth'), map_location=device)
model = checkpoint['model'].to(device).eval()
x_mean, x_std = checkpoint['x_mean'], checkpoint['x_std']
y_mean, y_std = checkpoint['y_mean'], checkpoint['y_std']

checkpoint2 = torch.load(('tf_mlp.pth'), map_location=device)
model2 = checkpoint2['model'].to(device).eval()
x_mean2, x_std2 = checkpoint2['x_mean'], checkpoint2['x_std']
y_mean2, y_std2 = checkpoint2['y_mean'], checkpoint2['y_std']

#checkpoint2 = torch.load(('control_mlp_w.pth'), map_location=device)
#model2 = checkpoint['model'].to(device).eval()
#x_mean2, x_std2 = checkpoint2['x_mean'], checkpoint2['x_std']
#y_mean2, y_std2 = checkpoint2['y_mean'], checkpoint2['y_std']


ix = 3
iu = 2
ih = 2
N = 30
#tf = 3

max_iter = 20

#%%
mu=0
sigma=0.5


xi = np.zeros(3)
xi[0] = -1.0 +np.random.normal(mu,sigma)
xi[1] = -2.0 +np.random.normal(mu,sigma)
xi[2] = 0 

xf = np.zeros(3)
xf[0] = 2.0 +np.random.normal(mu,sigma)
xf[1] = 2.0 +np.random.normal(mu,sigma)
xf[2] = 0

myModel = UnicycleModel.unicycle('Hello',ix,iu,'numeric_central')
myCost = UnicycleCost.unicycle('Hello',ix,iu,N)
myConst = UnicycleConstraints.UnicycleConstraints('Hello',ix,iu)

x0 = np.zeros((N+1,ix))
for i in range(N+1) :
    x0[i] = (N-i)/N * xi + i/N * xf
xini = np.hstack((xi,xf))
#x00 = np.vstack((x0[:, 0], x0[:, 1], x0[:, 2])).reshape(-1)

def normalize_input(sample, mean, std):
    std_safe = np.where(std == 0, 1.0, std)
    normed = (sample - mean) / std_safe
    return np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)

initial_norm = normalize_input(xini, x_mean, x_std)
initial_norm = torch.tensor(initial_norm, dtype=torch.float32).unsqueeze(0).to(device)

def unnormalize_output(pred, mean, std):
    std_safe = np.where(np.isnan(std), 1.0, std)
    return np.nan_to_num(pred * std_safe + mean)

with torch.no_grad():
    pred_v = model(initial_norm).cpu().numpy().squeeze()
    #pred_w = model2(initial_norm).cpu().numpy().squeeze()
    pred_tf = model2(initial_norm).cpu().numpy().squeeze()
pred_v = unnormalize_output(pred_v, y_mean, y_std)
#pred_w = unnormalize_output(pred_w, y_mean2, y_std2)
pred_v = pred_v.reshape(31,1)
#pred_w = pred_w.reshape(31,1)
temp = np.zeros((N+1,1))
#plt.figure()
#plt.plot(pred_v)
#plt.show()
#plt.figure()
#plt.plot(pred_w)
#plt.show()
u0 = np.concatenate((pred_v,temp), axis=1)
#u0 = u0.T
#plt.figure()
#plt.plot(u0[:,0])
#plt.show()

#plt.figure()
#plt.plot(u0[:,1])
#plt.show()

pred_tf = unnormalize_output(pred_tf, y_mean2, y_std2)

tf=pred_tf
delT = tf/N
#%%

i1 = PTR('unicycle',N,tf,max_iter,myModel,myCost,myConst,type_discretization="zoh",
          w_c=1,w_vc=1e3,w_tr=1e-1,w_rate=0,
         tol_vc=1e-6,tol_tr=1e-3)
x,u,xbar,ubar,total_num_iter,flag_boundary,l,l_vc,l_tr,x_traj,u_traj,T_traj = i1.run(x0,u0,xi,xf)

end_time = time.time()

print(f"실행 시간: {end_time - start_time:.4f}초")

#%%
start_time = time.time()
u00 = np.zeros((N+1,iu))
xo,uo,xbaro,ubaro,total_num_iter,flag_boundary,l,l_vc,l_tr,x_traj,u_traj,T_traj = i1.run(x0,u00,xi,xf)
end_time = time.time()

print(f"실행 시간: {end_time - start_time:.4f}초")
#%%

t_index = np.array(range(N+1))*delT
fS = 18

#plt.figure(figsize=(10,10))
#plt.subplot(221)
plt.figure()
plt.plot(x[:,0], x[:,1],'r:', linewidth=2.0, label = 'DNN_warmstart',zorder = 10)
plt.plot(xo[:,0], xo[:,1],'b-', linewidth=2.0, label = 'original_scp', zorder =5)
#plt.plot(xbar[:,0], xbar[:,1],'-.', linewidth=2.0)
plt.plot(xf[0],xf[1],"o",label='goal')
plt.plot(xi[0],xi[1],"o",label='start')
#plt.plot(xbaro[:,0], xbaro[:,1],'-', linewidth=2.0)
plt.gca().set_aspect('equal', adjustable='box')
plt.axis([-3, 3, -3, 3])
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend()
plt.grid()
plt.show()

#plt.subplot(222)
plt.figure()
plt.plot(t_index, xbar[:,0],'r:', linewidth=2.0,label='DNN_warmstart',zorder = 10)
plt.plot(t_index, xbaro[:,0],'b-', linewidth=2.0,label='original_scp', zorder =5)
plt.xlabel('time (s)')
plt.ylabel('X (m)')
plt.legend()
plt.grid()
plt.show()

#plt.subplot(223)
plt.figure()
plt.plot(t_index, xbar[:,1], 'r:', linewidth=2.0,label='DNN_warmstart',zorder = 10)
plt.plot(t_index, xbaro[:,1], 'b-', linewidth=2.0,label='original_scp', zorder =5)
plt.xlabel('time (s)')
plt.ylabel('Y (m)')
plt.legend()
plt.grid()
plt.show()

#plt.subplot(224)
plt.figure()
plt.plot(t_index, xbar[:,2], 'r:', linewidth=2.0,label='DNN_warmstart',zorder = 10)
plt.plot(t_index, xbaro[:,2], 'b-', linewidth=2.0,label='original_scp', zorder =5)
plt.xlabel('time (s)')
plt.ylabel('Heading angle(rad)')
plt.legend()
plt.grid()
plt.show()

plt.figure()
#plt.subplot(121)
if i1.type_discretization == "zoh" :
    plt.step(t_index, [*u[:N,0],u[N-1,0]], 'r--',alpha=1.0,where='post',linewidth=2.0,label='DNN_warmstart',zorder = 10)
    plt.step(t_index, [*uo[:N,0],uo[N-1,0]], 'b-',alpha=1.0,where='post',linewidth=2.0,label='original_scp', zorder =5)
elif i1.type_discretization == "foh" :
    plt.plot(t_index, u[:,0],'r--', linewidth=2.0,label='DNN_warmstart',zorder = 10)
    plt.plot(t_index, u[:,0],'b-', linewidth=2.0,label='original_scp', zorder =5)
plt.xlabel('time (s)')
plt.ylabel('v (m/s)')
plt.grid()
plt.legend()
plt.show()

#plt.subplot(122)
plt.figure()
if i1.type_discretization == "zoh" :
    plt.step(t_index, [*ubar[:N,1],ubar[N-1,1]], 'r--',alpha=1.0,where='post',linewidth=2.0,label='DNN_warmstart',zorder = 10)
    plt.step(t_index, [*ubaro[:N,1],ubaro[N-1,1]], 'b-',alpha=1.0,where='post',linewidth=2.0,label='original_scp', zorder =5)
elif i1.type_discretization == "foh" :
    plt.plot(t_index, ubar[:,1],'r--', linewidth=2.0,label='DNN_warmstart',zorder = 10)
    plt.plot(t_index, ubaro[:,1],'b-', linewidth=2.0,label='original_scp', zorder =5)
plt.xlabel('time (s)')
plt.ylabel('w (rad/s)')
plt.grid()
plt.legend()
plt.show()

print(x)
print(tf)
print(xi)
print(xf)
