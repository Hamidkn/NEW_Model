import numpy as np
import matplotlib as plt
from acting_force import acting_force
from rung_kutha import rung_kutha


def force():
    pass

def dxdt():
    pass

def run(self, init,tfinal,tinit,step,k,m,g):
    force = []
    steps = 1
    forcestep = 1
    phasestep = 1

    pass

m = 0.0362 * 1e-3
g = 9.81
k = 0.213
init = [0, 0, 0, 0]
ti = 0
tf = 0.064
step = 2e3

[t, s,v, force] = run(init,tf,ti,step,k,m,g)

[t, sv] = rung_kutha(force(), ti,tf, init,step)