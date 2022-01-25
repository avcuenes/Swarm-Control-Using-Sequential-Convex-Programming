import numpy as np
from cvxpy import *

class convex_collison:
  def __init__(self, x_init: list, y_init: list, x_final: list, y_final: list, N: int, numUAV: int, h: float, r_col: float):
    """
    This class create safe rota for swarm agent with respect to initial and final position of agent,
    number of agent and collision radius

    :param x_init: İnitial x position of agents
    :param y_init: İnitial y position of agents
    :param x_final: Final x position of agents
    :param y_final: Final y position of agents
    :param N: waypoint number
    :param numUAV: Number of agent
    :param h: discrete time
    :param r_col: Collision radius

    :return:Safe trajectory to agent
    """
    self.xc = x_init
    self.yc = y_init
    self.xt = x_final
    self.yt = y_final
    self.N = N
    self.numUAV = numUAV
    self.h = h
    self.r_col = r_col


  def main(self):
    """
    This function loop main function of algorithm

    :return: Safe waypoint for swarm
    """

    trajUAVPre = []
    trajUAVNext = []
    for i in range(self.numUAV):
      previous_x_position, previous_y_position  = self.optimum_trajectory_initialize(xc=self.xc[i], yc=self.yc[i], xt=self.xt[i], yt=self.yt[i], N=self.N, h=self.h)
      trajUAVPre.append([previous_x_position, previous_y_position])
    K = np.arange(self.numUAV) # Total number of uav
    trajUAVNext = trajUAVPre
    while len(K) > 0:

      for i in K:
        ax1 , ay1 = self.optimum_trajectory_regenerate(trajUAV=trajUAVPre, xc=self.xc[i], yc=self.yc[i], xt=self.xt[i], yt=self.yt[i], N=self.N, h=self.h, r=self.r_col, ID=i,K=K)
        trajUAVNext[i] = [ax1,ay1]
      
      for i in range(0, self.numUAV):
        flag = 1
        for j in range(i+1, self.numUAV):
            for k in range(0, self.N+1):
                var1 = np.array([[trajUAVNext[i][0][k], trajUAVNext[i][1][k]]])
                var2 = np.array([[trajUAVPre[i][0][k], trajUAVPre[i][1][k]]])
                var3 = np.array([[trajUAVNext[j][0][k], trajUAVNext[j][1][k]]])
                if not (np.linalg.norm(var1-var2, np.inf) < 1 and np.linalg.norm((var1-var3), 2) >= self.r_col):
                    flag = 0

                if j == self.numUAV-1 and k == self.N and flag == 1:
                    if i in K:

                      K = np.delete(K, np.where(K == i))
                      trajUAVPre[i] = trajUAVNext[i]    
      if len(K) == 1:K = []

    return trajUAVNext

  def optimum_trajectory_initialize(self, xc: list, yc: list, xt: list, yt: list, N: int, h: float):
      """
      This function calculate short and dynamicly feasible path for agents

      :xc: İnitial x position of agents
      :yc: İnitial y position of agents
      :xt: Final x positon of agents
      :yt: Final y position of agents
      :N: number of waypoint
      :h: discerte time

      :return: path to target
      """
      n = N
      amax = 5

      
      x = []
      v = []
      a = []
      jerk = []
      for i in range(n + 2):
          x += [Variable(2)]
          v += [Variable(2)]
          a += [Variable(2)]
          jerk += [Variable(2)]
      

      
      constr = [x[0][0] == xc   ,  x[0][1] == yc    ,    
                x[n][0] == xt   ,  x[n][1] == yt    ]
      cost = 0
      for i in range(n+2):
        cost += norm(jerk[i],1)

      c = np.eye(2)
      
      for i in range(0,n+1):
          constr += [#norm(x[i] - x[i + 1]) <= L / n,
                     v[i+1] == v[i] + h*a[i],
                     x[i+1] == x[i] + h*v[i]+0.5*a[i]*h*h,
                     jerk[i] == (a[i+1]-a[i])/h,
                     a[i]<=amax,
                     a[i]>=-amax,
                     jerk[i]<=2.5,
                     ]        
          
      prob = Problem( Minimize(cost), constr)
      result = prob.solve(solver= ECOS,warm_start=True)
      ax1 = []
      ay1 = []
      for k in range(0,n+1):
          ax1.append(x[k][0].value)
          ay1.append(x[k][1].value)
      return ax1, ay1

  def optimum_trajectory_regenerate(self,trajUAV: list, xc: list, yc: list, xt:list, yt:list, N: int, h: float, r:float, ID:int, K:list):
      """
      This function calculate safe path and dynamicly feasible path for agents

      :xc: İnitial x position of agents
      :yc: İnitial y position of agents
      :xt: Final x positon of agents
      :yt: Final y position of agents
      :N: number of waypoint
      :h: discerte time
      :r: collision radius
      :ID: ID number of agent
      :K: list of vehicle
      :return: safe path to target
      """
      n = N
      amax = 5

      xx = []
      vx = []
      ax = []

      jerkx = []
      for i in range(n + 2):
          xx += [Variable(2)]
          vx += [Variable(2)]
          ax += [Variable(2)]
          jerkx += [Variable(2)]
          
      
      #ax= Variable(1)
      c = np.eye(2)
      cost1 = 0
      for b in range(n+2):
        cost1 += norm(jerkx[b],1)
      
      constr1 = [xx[0][0] == xc,  xx[0][1] == yc, xx[n][0] == xt,  xx[n][1] == yt]
      
          
      for k in range(0,n+1):
        constr1 += [#norm(xx[k+1] - xx[k]) <= Lx / n,
                xx[k+1] == xx[k] + h*vx[k] +h*h*0.5*ax[k], 
                vx[k+1] == vx[k] + h*ax[k],
                jerkx[k] == (ax[k+1]-ax[k])/h,
                ax[k] >= -amax,
                ax[k] <= amax,
                jerkx[k] <= 2.5]
        for j in list(set(K)-set([ID])):
            var1 = np.array([[trajUAV[ID][0][k], trajUAV[ID][1][k]]])
            var2 = np.array([[trajUAV[j][0][k], trajUAV[j][1][k]]])
            var3 = (var1.T-var2.T).T@c.T
            var4 = var3@c
            var5 = var4@(reshape(xx[k],(2, 1))-var2.T)
            constr1 += [var5>=(r*pnorm(c@(var1.T-var2.T), 2))]


      prob1 = Problem(Minimize(cost1), constr1)


      result = prob1.solve(solver= ECOS, warm_start=True)


      ax1 = []
      ay1 = []
      for k in range(0, n+1):
        ax1.append(xx[k][0].value)
        ay1.append(xx[k][1].value)

      return ax1, ay1
       
            
      





