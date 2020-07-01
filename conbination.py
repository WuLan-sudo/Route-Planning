from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
import dccp
import math

np.random.seed(0)

T = 100
l = 6.0
m = 1
v_max = 0.15
d_min = 0.6
A = np.matrix([[1,0,0.1,0],
     [0,1,0,0.1],
     [0,0,0.95,0],
     [0,0,0,0.95]])
B = np.matrix([[0,0],
     [0,0],
     [0.1/float(m),0],
     [0,0.1/float(m)]])
C = np.matrix([[1,0,0,0],
     [0,1,0,0]])
d = 2 #dim of space
n = 2 #number of systems
x_ini = [[1,4,0.5,0.5],[3,5,-0.5,0]]
x_end = [[3,3,0,-0.5],[3.5,3.5,0.5,-0.5]]
f_max = 0.5
##Obstacle list
listx = [2,3]
listy = [4.0,4.5]
r = [0.2,0.1]
d2_min = 0.1
p = np.matrix([listx,listy])

u = []
y = []
x = []
for i in range(n):
    u.append([])
    y.append([])
    x.append([])
cost = 0
constr = []

for i in range(n):
    for t in range(T):
        u[i] += [Variable(d)]
        constr += [norm(u[i][-1],'inf') <= f_max]
        cost += pnorm(u[i][-1],1)
        y[i] += [Variable(d)]
        x[i] += [Variable(2*d)]
        constr += [y[i][-1] == C*x[i][-1]]
    constr += [x[i][0] == x_ini[i]]
    constr += [x[i][-1] == x_end[i]]
for i in range(n):
    for t in range(T-1):
        constr += [x[i][t+1] == A*x[i][t] + B*u[i][t]]
        ##avoid obstacles
    for t in range(T):
        for n in range(len(listx)):
            constr += [norm(y[i][t]-p[:,n]) >= r[n] + d2_min]

for t in range(T):
    for i in range(n-1):
        for j in range(i+1,n):
            constr += [norm(y[i][t] - y[j][t]) >= d_min]


print("start solving")
prob = Problem(Minimize(cost), constr)
prob.solve(method='dccp', ep = 1e-1)
print("finish!")

##Drawer
def circle_draw(x0,y0,r):
    circ = np.linspace(0,2*math.pi,50)
    x = []
    y = []
    for i in circ:
        x.append(x0 + r*math.cos(i))
        y.append(y0 + r*math.sin(i))
    plt.plot(x, y,'r')
plt.figure(figsize=(20,5))
# obstacle draw
for i in range(len(listx)):
    circle_draw(listx[i],listy[i],r[i])
# plt.subplot(132)
ax = [xx.value[0] for xx in y[0]]
ay = [xx.value[1] for xx in y[0]]
plt.plot(ax, ay,'b-')
plt.quiver(x_ini[0][0], x_ini[0][1], x_ini[0][2], x_ini[0][3],
           units='xy', scale=2, zorder=3, color='black',
           width=0.01, headwidth=4., headlength=5.)
plt.quiver(x_ini[1][0], x_ini[1][1], x_ini[1][2], x_ini[1][3],
           units='xy', scale=2, zorder=3, color='black',
           width=0.01, headwidth=4., headlength=5.)
plt.quiver(x_end[0][0], x_end[0][1], x_end[0][2], x_end[0][3],
           units='xy', scale=2, zorder=3, color='black',
           width=0.01, headwidth=4., headlength=5.)
plt.quiver(x_end[1][0], x_end[1][1], x_end[1][2], x_end[1][3],
           units='xy', scale=2, zorder=3, color='black',
           width=0.01, headwidth=4., headlength=5.)
bx = [xx.value[0] for xx in y[1]]
by = [xx.value[1] for xx in y[1]]
plt.plot(bx, by,'b--')
plt.axis('equal')
plt.xlim(0.5,4.5)
plt.ylim(2,6)
plt.show()