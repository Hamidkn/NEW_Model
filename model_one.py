import numpy as np
import matplotlib.pyplot as plt
from acting_force import acting_force


dt=1e-5
t = np.arange(0, 1.1, dt)
u=len(t)
g = 9.81
k = 0.213
R=100
L=1.3e-3
m=(0.0075 + 5 * 0.006);
step=5*1e3
Sstep= 4*step

x1 = []
x2 = []
curr2 = []
curr1 = []
Ftotal = []
x1 = np.zeros(int(u))
x2 = np.zeros(int(u))
curr1 = np.zeros(int(u))
curr2 = np.zeros(int(u))
Ftotal = np.zeros(int(u))
j = 0

for i in range(u - 1):
    x1[i+1] = dt*x2[i] + x1[i]
    if (x1[i] < 0.0005):
        phase = 1
        curr = curr1[i]
        curr2[i]=0
        v1=18
        v2=0
        Fx = acting_force(x1[i], phase)
        Ftotal[i] = Fx
        # if Ftotal[i] == 0 :
        #     continue
    elif ( 0.0005 < x1[i] and x1[i] < 0.001):
        # print('Phase 2 ....')
        phase = 2
        curr = curr2[i]
        curr1[i] = 0
        v1 = 0
        v2 = 18
        Fx = acting_force(x1[i], phase)
        Ftotal[i] = Fx
    #     if Ftotal[i] == 0:
    #         continue
    elif (0.001 < x1[i] and x1[i] < 0.0015):
        # print('Phase 3 ....')
        phase = 3
        curr = curr1[i]
        curr2[i] = 0
        v2 = 0
        v1 = -18
        Fx = acting_force(x1[i], phase)
        Ftotal[i] = Fx
    #     if Ftotal[i] == 0:
    #         continue
    elif (0.0015 < x1[i] and x1[i] < 0.002):
        # print('phase 4 .....')
        phase = 4
        curr = curr2[i]
        curr1[i] = 0
        v1 = 0
        v2 = -18
        Fx = acting_force(x1[i], phase)
        Ftotal[i] = Fx
    # else:
    #     break

    curr1[i+1]= dt*(((-R/L)*curr1[i]) + v1*(1/L)) + curr1[i]
    curr2[i+1]= dt*(((-R/L)*curr2[i]) + v2*(1/L)) + curr2[i]
    x2[i+1]= (dt*Ftotal[i]*(curr/0.1))/m + x2[i]
    j = i

# x1 = x1[1:j+1]
# x2 = x2[1:j+1]
# t = t[1:j+1]
# Ftotal = Ftotal[1:j+1]
# curr1 = curr1[1:j+1]
# curr2 = curr2[1:j+1]

# plt.plot(t,x1)
# plt.show()

# plt.plot(x1,Ftotal)
# plt.show()


# plt.plot(x1,curr1)
# plt.plot(x1,curr2)
# plt.show()

# plt.plot(t,x2)
# plt.show()

inp = [curr1, curr2, Ftotal, x2]
out = [x1]
print(np.shape(inp))