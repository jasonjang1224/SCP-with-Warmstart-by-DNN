from __future__ import division
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import numpy as np
import cvxpy as cvx
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))

import cost
import model
import IPython
from PTR import PTR

from Scaling import TrajectoryScaling

class PTR_tf_free(PTR):
    def __init__(self,name,horizon,tf,maxIter,Model,Cost,Const,Scaling=None,type_discretization='zoh',
        w_c=1,w_vc=1e4,w_tr=1e-3,w_rate=0,
        tol_vc=1e-10,tol_tr=1e-3,tol_bc=1e-3,
        flag_policyopt=False,verbosity=True):
        
        super().__init__(name,horizon,tf,maxIter,Model,Cost,Const,Scaling,type_discretization,
                         w_c,w_vc,w_tr,w_rate,tol_vc,tol_tr,tol_bc,flag_policyopt,verbosity)

    def cvxopt(self):
        # TODO - we can get rid of most of loops here 
        
        # state & input & horizon size
        ix = self.model.ix
        iu = self.model.iu
        N = self.N

        if self.flag_update_scale is True :
            self.Scaling.update_scaling_from_traj(self.x,self.u)
        Sx,iSx,sx,Su,iSu,su = self.Scaling.get_scaling()

        S_sigma = self.Scaling.S_sigma

        x_cvx = cvx.Variable((N+1,ix))
        u_cvx = cvx.Variable((N+1,iu))
        vc = cvx.Variable((N,ix))
        sigma = cvx.Variable(nonneg=True)

        # initial & final boundary condition
        constraints = []
        constraints.append(Sx@x_cvx[0]+sx  == self.xi)
        constraints.append(Sx@x_cvx[-1]+sx == self.xf)

        # state and input contraints
        for i in range(0,N+1) :
            h = self.const.forward(Sx@x_cvx[i]+sx,Su@u_cvx[i]+su,self.x[i],self.u[i],i==N)
            constraints += h

        # model constraints
        for i in range(0,N) :
            if self.type_discretization == 'zoh' :
                constraints.append(Sx@x_cvx[i+1]+sx == self.A[i]@(Sx@x_cvx[i]+sx)+self.B[i]@(Su@u_cvx[i]+su)
                                                                            +sigma*S_sigma*self.s[i]
                                                                            +self.z[i]
                                                                            +vc[i])
            elif self.type_discretization == 'foh' :
                constraints.append(Sx@x_cvx[i+1]+sx == self.A[i]@(Sx@x_cvx[i]+sx)+self.Bm[i]@(Su@u_cvx[i]+su)
                                                                            +self.Bp[i]@(Su@u_cvx[i+1]+su)
                                                                            +sigma*S_sigma*self.s[i]
                                                                            +self.z[i]
                                                                            +vc[i])

        # cost
        objective = []
        objective_vc = []
        objective_tr = []
        objective_rate = []

        # ======================== Start of Modified Section ========================
        # 1. Separate the main costs
        
        # 1a. Cost associated with the final time (tf) (for a minimum time problem, the cost is tf itself)
        time_cost = sigma * S_sigma
        objective.append(time_cost)

        # 1b. Cost associated with state (x) and input (u) (summed over the entire trajectory)
        # estimate_cost_cvx in UnicycleCost.py takes x and u as arguments.
        stage_cost = 0
        for i in range(N + 1):
            # Correctly pass the state (x_cvx[i]) and input (u_cvx[i]) at each time step.
            # It's standard practice to pass the scaled values, as in the original PTR code.
            stage_cost += self.cost.estimate_cost_cvx(Sx@x_cvx[i] + sx, Su@u_cvx[i] + su)

        objective.append(self.w_c * stage_cost)
        # ======================== End of Modified Section ==========================
        
        # 2. Virtual Control and Trust Region costs
        for i in range(0,N+1) :
            if i < N :
                objective_vc.append(self.w_vc * cvx.norm(vc[i],1))
                objective_rate.append(self.w_rate * cvx.quad_form(u_cvx[i+1]-u_cvx[i],np.eye(iu)))
            
            objective_tr.append( self.w_tr * (cvx.quad_form(x_cvx[i] -
                iSx@(self.x[i]-sx),np.eye(ix)) +
                cvx.quad_form(u_cvx[i]-iSu@(self.u[i]-su),np.eye(iu))) )
        
        # 3. Trust Region cost for the final time (tf)
        objective_tr.append(self.w_tr*(sigma-self.tf/S_sigma)**2)

        l = cvx.sum(objective)
        l_vc = cvx.sum(objective_vc)
        l_tr = cvx.sum(objective_tr)
        l_rate = cvx.sum(objective_rate)

        l_all = l + l_vc + l_tr + l_rate
        prob = cvx.Problem(cvx.Minimize(l_all), constraints)

        # (The rest of the code is the same as the original)
        error = False
        prob.solve(verbose=False,solver=cvx.ECOS)

        if prob.status == cvx.OPTIMAL_INACCURATE :
            print("WARNING: inaccurate solution")

        try :
            x_bar = np.zeros_like(self.x)
            u_bar = np.zeros_like(self.u)
            for i in range(N+1) :
                x_bar[i] = Sx@x_cvx[i].value + sx
                u_bar[i] = Su@u_cvx[i].value + su
            sigma_bar = sigma.value * S_sigma
        except (ValueError, TypeError) :
            print(prob.status,"FAIL: solution not found")
            error = True
            x_bar, u_bar, sigma_bar, vc_val = None, None, None, None
        
        # Add exception handling since vc.value can be None
        vc_val = vc.value if vc.value is not None else np.zeros_like(self.vc)

        return prob.status,l.value,l_vc.value,l_tr.value,x_bar,u_bar,sigma_bar,vc_val,error

    def run(self,x0,u0,xi,xf):
        # initial trajectory
        self.x0 = x0

        # initial input
        self.u0 = u0
        self.u = u0
        self.x = x0
        # initial condition
        self.xi = xi
        # final condition
        self.xf = xf
        
        # save trajectory
        x_traj, u_traj, T_traj = [], [], []
        
        # Set initial cost values
        self.c = 1e3 # The initial cost is set to a sufficiently large value
        self.cvc = 0
        self.ctr = 0

        total_num_iter = 0
        flag_boundary = False
        
        # Start of the optimization loop
        for iteration in range(self.maxIter) :
            # 1. Linearize the dynamics model
            # Obtain the linear model matrices A, B, s, z using the current iteration's x, u, tf, and delT
            self.A, self.B, self.Bm, self.Bp, self.s, self.z, _, _ = self.get_linearized_matrices(self.x, self.u, self.delT, self.tf)

            # 2. Convex Optimization (SCP step)
            # Call the overridden cvxopt to optimize x, u, and also tf
            prob_status, l, l_vc, l_tr, x_new, u_new, tf_new, vc_new, error = self.cvxopt()

            if error:
                print("Error in CVXOPT step. Terminating.")
                total_num_iter = self.maxIter # Set to max iterations if an error occurs
                break

            # 3. Forward Simulation (trajectory propagation)
            # Generate the trajectory along the actual nonlinear dynamics model using the new u_new and tf_new
            self.xfwd, self.ufwd = self.forward_multiple(x_new, u_new, tf_new, iteration)

            # Check dynamics constraint satisfaction (boundary condition check)
            bc_error_norm = np.max(np.linalg.norm(self.xfwd - x_new, axis=1))
            flag_boundary = bc_error_norm < self.tol_bc

            # 4. Update results
            # Reduction between the cost of the previous step and the predicted cost of the current step
            reduction = (self.c + self.cvc + self.ctr) - (l + l_vc + l_tr)

            # Update with the newly calculated trajectory and cost
            self.x = x_new
            self.u = u_new
            self.vc = vc_new
            self.c = l 
            self.cvc = l_vc 
            self.ctr = l_tr
            
            # --- The most important update part ---
            self.tf = tf_new
            self.delT = self.tf / self.N
            # --------------------------------

            x_traj.append(self.x)
            u_traj.append(self.u)
            T_traj.append(self.tf)

            # 5. Print log
            if self.verbosity == True:
                if self.last_head:
                    print("iteration   total_cost        cost        ||vc||     ||tr||       reduction   w_tr        dynamics")
                    self.last_head = False
                print("%-12d%-18.3f%-12.3f%-12.3g%-12.3g%-12.3g%-12.3f%-1d(%2.3g)" % (
                    iteration + 1, self.c + self.cvc + self.ctr,
                    self.c, self.cvc / self.w_vc, self.ctr / self.w_tr,
                    reduction, self.w_tr, flag_boundary, bc_error_norm))
            
            # 6. Check termination conditions
            if flag_boundary and self.ctr / self.w_tr < self.tol_tr and self.cvc / self.w_vc < self.tol_vc:
                if self.verbosity:
                    print("SUCCESS: Virtual control and trust region are within tolerance.")
                total_num_iter = iteration + 1
                break

        if iteration == self.maxIter - 1:
            print("WARNING: Reached maximum iterations.")
            total_num_iter = self.maxIter

        return self.xfwd, self.ufwd, self.x, self.u, self.tf, total_num_iter, flag_boundary, l, l_vc, l_tr, x_traj, u_traj, T_traj