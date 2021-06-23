
R=100
L=1.3e-3

def current(dt,v1,v2,current1,current2):
    R=100
    L=1.3e-3

    curr1= dt*(((-R/L)*current1) + v1*(1/L)) + current1
    curr2= dt*(((-R/L)*current2) + v2*(1/L)) + current2
    
    return curr1, curr2