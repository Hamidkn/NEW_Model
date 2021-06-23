def rung_kutha(tinit,tfinal,init,step):
    h = (tfinal - tinit)/step
    y = [init]
    x = [tinit]

    for i in range(step):
        k1 = func(x[i], y[i,:])
        k2 = func(x[i] + h/2, y[i,:] + (h*k1)/2)
        k3 = func(x[i] + h/2, y[i,:] + (h*k2)/2)
        k4 = func(x[i] + h, y[i,:] + h*k3)

        y[i+1, :] = y[i,:] + h*(k1 + 2 * k2 + 2 * k3 + k4) / 60
        x[i+1] = x[i] + h

