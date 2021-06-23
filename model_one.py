import numpy as np
from motion import motion_profile
from plot import draw_plots


dt=1e-5
t = np.arange(0, 1.1, dt)
u=len(t)
g = 9.81
k = 0.213
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

x1, x2, Ftotal, curr1, curr2 = motion_profile(dt, u, x1, x2, Ftotal, curr1, curr2)

draw_plots(t, x1, x2, Ftotal, curr1, curr2)

inp = [curr1, curr2, Ftotal, x2]
out = [x1]
